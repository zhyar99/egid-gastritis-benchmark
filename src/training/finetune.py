"""Optional end-to-end fine-tuning (backbone gradients on), MPS-friendly.

Images come from the uint8 256-px cache (never the originals), are augmented
on the fly (training only) and encoded by a trainable timm backbone. After
training, the fine-tuned encoder embeds every image once and the shared
evaluation path (`trainer.finish`) produces the same calibrated, missing-view
and record outputs as the frozen-feature runs.
"""
from __future__ import annotations

import copy
import logging
import time

import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2 as T

from ..common import mps_oom_hint, seed_everything
from ..data.dataset import Cohort, pad_bags, sample_view
from ..data.metadata import TASK_KEYS
from ..data.preprocessing import open_cache
from ..data.splits import load_split
from ..models.baseline import ImageLevelModel
from ..models.core_mil import MILModel
from ..models.encoders import Normalize, create_encoder
from ..models.structured_head import derive_mapping, finalize
from .losses import class_weights, cross_view_consistency, supervised_loss, warmup
from .trainer import Runner, RunSpec, config_hash, finish, variant_config

log = logging.getLogger(__name__)


def transforms(size: int, train: bool):
    if not train:
        return T.Resize((size, size), antialias=True)
    # moderate geometry/photometry; no hue shift (mucosal colour is diagnostic)
    return T.Compose([T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.11), antialias=True),
                      T.RandomHorizontalFlip(), T.RandomRotation(15),
                      T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)])


class FineTuner:
    def __init__(self, cfg: dict, co: Cohort, spec: RunSpec, device: torch.device, profile: dict):
        self.cfg, self.co, self.spec, self.device = cfg, co, spec, device
        self.ft = cfg["finetune"]
        self.v = variant_config(cfg, spec.variant, profile)
        self.images = open_cache(cfg["paths"]["cache"], self.ft["cache_size"], co.index)
        sp = load_split(cfg["paths"]["splits"], spec.split)
        self.ids = dict(train=sp["train"], val=sp["val"], test=sp["test"])
        seed_everything(spec.seed)
        self.rng = np.random.default_rng(spec.seed)
        if profile.get("max_train_patients"):
            self.ids["train"] = sorted(self.rng.choice(self.ids["train"], profile["max_train_patients"], replace=False))
        enc, info = create_encoder(spec.encoder, cfg["encoders"])
        self.info = info
        self.y_train = co.y(self.ids["train"])
        self.mapping_np = derive_mapping(self.y_train, co.n_classes)
        v = self.v
        if v.get("head") == "structured" and self.mapping_np is None:
            log.warning("%s: mapping not deterministic on training patients; using independent heads", spec.run_id)
            v["head"] = "independent"
        if v["kind"] == "image":
            head = ImageLevelModel(info["dim"], co.n_classes, hidden=v["hidden"], dropout=v["dropout"])
        else:
            head = MILModel(info["dim"], co.n_classes, pooling=v["pooling"], head=v["head"], mapping=self.mapping_np,
                            hidden=v["hidden"], attn_dim=v["attn_dim"], dropout=v["dropout"], in_norm=True)
        params = list(enc.named_parameters())
        n_frozen = int(round((1 - v.get("backbone_trainable_frac", 1.0)) * len(params)))
        for _, p in params[:n_frozen]:  # freeze the earliest layers if requested
            p.requires_grad_(False)
        self.enc, self.head = enc.to(device), head.to(device)
        self.norm = Normalize(info["mean"], info["std"]).to(device)
        self.opt = torch.optim.AdamW([dict(params=[p for p in enc.parameters() if p.requires_grad], lr=v["lr_backbone"]),
                                      dict(params=head.parameters(), lr=v["lr"])], weight_decay=v["weight_decay"])
        self.weights = class_weights(self.y_train, co.n_classes)
        self.aug, self.plain = transforms(self.ft["image_size"], True), transforms(self.ft["image_size"], False)

    def _batch(self, rows: np.ndarray, train: bool) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(self.images[np.sort(rows)])).permute(0, 3, 1, 2)
        order = np.argsort(np.argsort(rows))  # memmap reads in sorted order; restore requested order
        x = x[torch.as_tensor(order)]
        tf = self.aug if train else self.plain
        x = torch.stack([tf(im) for im in x]) if train else tf(x)
        return self.norm(x.permute(0, 2, 3, 1).to(self.device))

    @torch.no_grad()
    def embed(self, rows: np.ndarray) -> torch.Tensor:
        self.enc.eval()
        out = [self.enc(self._batch(rows[s: s + self.ft["eval_chunk"]], False)) for s in
               range(0, len(rows), self.ft["eval_chunk"])]
        self.enc.train()
        return torch.cat(out)

    def _sup(self, lp, y):
        return supervised_loss(lp, y, self.v["supervised_loss"], self.weights, self.v["focal_gamma"])[0]

    def _step(self, loss: torch.Tensor, k: int):
        (loss / self.ft["grad_accum"]).backward()
        if (k + 1) % self.ft["grad_accum"] == 0:
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

    def epoch_image(self) -> float:
        rows, pos = self.co.image_rows(self.ids["train"])
        perm = self.rng.permutation(len(rows))
        bs, losses = self.ft["batch_images"], []
        for k, s in enumerate(range(0, len(perm), bs)):
            b = perm[s: s + bs]
            lp = finalize(self.head(self.enc(self._batch(rows[b], True))), None)
            loss = self._sup(lp, torch.as_tensor(self.y_train[pos[b]], device=self.device))
            self._step(loss, k)
            losses.append(loss.item())
        return float(np.mean(losses))

    def epoch_mil(self, epoch: int) -> float:
        v, co, cons = self.v, self.co, self.cfg["consistency"]
        lam2 = cons["lambda"] * warmup(epoch, cons["warmup_start"], cons["warmup_epochs"]) if v["consistency"] else 0.0
        order = list(self.rng.permutation(self.ids["train"]))
        losses = []
        for k, s in enumerate(range(0, len(order), self.ft["batch_patients"])):
            pids = order[s: s + self.ft["batch_patients"]]
            # teacher view = random K images (memory cap); student view = a subset of it
            views = [co.bags[p][np.sort(self.rng.choice(len(co.bags[p]), min(self.ft["bag_size_train"],
                                                                              len(co.bags[p])), replace=False))]
                     for p in pids]
            flat = np.concatenate(views)
            f = self.enc(self._batch(flat, True))
            offs = np.cumsum([0] + [len(r) for r in views])
            rows_t = [np.arange(offs[i], offs[i + 1]) for i in range(len(views))]
            y = torch.as_tensor(co.y(pids), device=self.device)
            out_t = self.head(*pad_bags(f, rows_t))
            loss = self._sup(self.head.log_probs(out_t), y)
            if v["view_dropout"]:
                rows_s = [r[sample_view(len(r), self.rng, self.cfg["view_dropout"], "student")] for r in rows_t]
                out_s = self.head(*pad_bags(f, rows_s))
                loss = (loss + self._sup(self.head.log_probs(out_s), y)) / 2
                if v["consistency"]:
                    l_cv, _ = cross_view_consistency(out_t["logits"], out_s["logits"], cons["temperature"],
                                                     cons["confidence"])
                    loss = loss + lam2 * l_cv
            self._step(loss, k)
            losses.append(loss.item())
        return float(np.mean(losses))

    @torch.no_grad()
    def validate(self) -> dict:
        self.head.eval()
        pids = self.ids["val"]
        rows, pos = self.co.image_rows(pids)
        f = self.embed(rows)
        if self.v["kind"] == "image":
            base = ImageLevelModel.aggregate(self.head(f), torch.as_tensor(pos, device=self.device), len(pids))
            lp = finalize(base, None)
        else:
            offs = np.cumsum([0] + [len(self.co.bags[p]) for p in pids])
            lp = self.head.log_probs(self.head(*pad_bags(f, [np.arange(offs[i], offs[i + 1]) for i in range(len(pids))])))
        self.head.train()
        y = self.co.y(pids)
        nll = np.mean([-lp[t][np.arange(len(y)), y[:, j]].mean().item() for j, t in enumerate(TASK_KEYS)])
        acc = np.mean([(lp[t].argmax(1).cpu().numpy() == y[:, j]).mean() for j, t in enumerate(TASK_KEYS)])
        return dict(val_nll=float(nll), val_mean_acc=float(acc))

    def fit(self) -> dict:
        v = self.v
        sel = v.get("selection", "val_nll")
        best, bad, hist, t0 = None, 0, [], time.time()
        self.opt.zero_grad(set_to_none=True)
        for epoch in range(v["max_epochs"]):
            te = time.time()
            try:
                tr = self.epoch_image() if v["kind"] == "image" else self.epoch_mil(epoch)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise RuntimeError(mps_oom_hint(e)) from e
                raise
            vl = self.validate()
            score = vl["val_nll"] if sel == "val_nll" else -vl["val_mean_acc"]
            hist.append(dict(epoch=epoch, sup=tr, **vl, seconds=time.time() - te))
            log.info("%s epoch %d: train %.3f val_nll %.3f val_acc %.3f (%.0fs)", self.spec.run_id, epoch, tr,
                     vl["val_nll"], vl["val_mean_acc"], time.time() - te)
            if best is None or score < best[0] - 1e-4:
                best, bad = (score, epoch, copy.deepcopy(self.enc.state_dict()), copy.deepcopy(self.head.state_dict())), 0
            else:
                bad += 1
                if bad >= v["patience"]:
                    break
        self.enc.load_state_dict(best[2])
        self.head.load_state_dict(best[3])
        return dict(best_epoch=best[1], epochs_run=len(hist), best_val_nll=min(h["val_nll"] for h in hist),
                    selection=sel, train_seconds=time.time() - t0, history=hist,
                    encoder_params=sum(p.numel() for p in self.enc.parameters()))


def run_finetune(cfg: dict, co: Cohort, spec: RunSpec, device: torch.device, profile: dict, profile_name: str,
                 force: bool = False) -> dict | None:
    P = cfg["paths"]
    rec_p = P["outputs"] / "runs" / f"{spec.run_id}.json"
    chash = config_hash(cfg, spec, profile)
    if rec_p.exists() and not force:
        log.info("skip %s (record exists)", spec.run_id)
        return None
    t0 = time.time()
    ftr = FineTuner(cfg, co, spec, device, profile)
    fit = ftr.fit()
    feats = ftr.embed(np.arange(len(co.index))).cpu().numpy()  # fine-tuned embeddings of every image
    P["checkpoints"].mkdir(parents=True, exist_ok=True)
    torch.save(ftr.enc.state_dict(), P["checkpoints"] / f"{spec.run_id}_encoder.pt")
    # evaluation reuses the frozen-feature path with the trained head (features already normalised by in_norm /
    # the paper's linear heads, so no extra standardisation)
    prof = {**profile, "overrides": {**profile.get("overrides", {}), "feature_standardize": False}}
    r = Runner(cfg, co, feats, spec, torch.device("cpu"), prof, P["outputs"])
    r.model = ftr.head.cpu().eval()
    return finish(cfg, r, fit, profile, profile_name, chash, t0)
