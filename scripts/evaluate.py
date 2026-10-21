"""Phase 8: aggregate completed runs into metrics, statistics and Tables 1-8 (CSV + LaTeX).

Only completed run records are used; missing seeds/folds are reported, never filled in.
  python scripts/evaluate.py --scheme cv      # development-cohort cross-validation
  python scripts/evaluate.py --scheme test    # untouched test cohort (after the protocol freeze)
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_config, setup_logging  # noqa: E402
from src.data.dataset import Cohort  # noqa: E402
from src.evaluation.aggregate import Pooled, confusion, load_records, paired, per_class, reliability_table, summarize  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
TASK_NAMES = dict(hp="H. pylori", atrophy="Atrophy presence", distribution="Atrophy distribution", type="Gastritis type",
                  mean="Mean of 4 tasks")
BASELINE_ENCODERS = dict(B1="resnet50", B2="efficientnet_b0", B3="densenet121")


def fmt(r, k=3):
    if pd.isna(r.get("estimate")):
        return "n/a"
    s = f"{r['estimate']:.{k}f}"
    if "lo" in r and not pd.isna(r.get("lo")):
        s += f" [{r['lo']:.{k}f}, {r['hi']:.{k}f}]"
    elif r.get("seed_sd", 0):
        s += f" ± {r['seed_sd']:.{k}f}"
    return s


def save(df: pd.DataFrame, name: str, out: Path, caption: str):
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    tex = df.to_latex(index=False, escape=True, na_rep="--", caption=caption, label=f"tab:{name}")
    (out / f"{name}.tex").write_text(tex)


if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="cv", choices=["cv", "test"])
    ap.add_argument("--n-boot", type=int, default=cfg["protocol"]["bootstrap"])
    ap.add_argument("--smoke", action="store_true", help="read the smoke-test output tree (pipeline check only)")
    a = ap.parse_args()
    setup_logging(f"evaluate_{a.scheme}")
    P = cfg["paths"]
    if a.smoke:
        P = cfg["paths"] = {**P, "outputs": P["outputs"] / "smoke"}
    co = Cohort.load(P["data"], P["audit"])
    recs = load_records(P["outputs"] / "runs")
    pred_dir = P["outputs"] / "predictions"
    tabs = P["outputs"] / "tables" / a.scheme
    mdir = P["outputs"] / "metrics" / a.scheme
    mdir.mkdir(parents=True, exist_ok=True)
    enc0 = cfg["protocol"]["primary_encoder"]
    fracs = cfg["protocol"]["missing_view_fractions"]
    recs.to_csv(mdir / "run_records.csv", index=False)

    # ---- which (label, variant, encoder) rows exist
    models = [(f"{b} IMG ({e})", "IMG", e) for b, e in BASELINE_ENCODERS.items()]
    models += [(v, v, enc0) for v in ["A0", "B4", "A1", "A2", "A3", "A4", "A5", "A6", "X1", "X2", "X3", "X4",
                                         "M2", "M3", "M5", "M6", "N1", "N2", "N3", "N4", "N5", "N6",
                                         "C1", "C2", "C3"]]
    for v in cfg["finetune_variants"]:
        for e in sorted(recs[recs.variant == v].encoder.unique()):
            models.append((f"{v} ({e})", v, e))
    pools = {}
    for label, v, e in models:
        p = Pooled(recs, pred_dir, v, e, 224, a.scheme)
        if p.ok:
            pools[label] = p
    status = [dict(model=l, variant=v, encoder=e, seeds=",".join(map(str, pools[l].seeds)) if l in pools else "",
                   complete=l in pools) for l, v, e in models]
    pd.DataFrame(status).to_csv(mdir / "completeness.csv", index=False)
    print(pd.DataFrame(status).to_string(index=False))
    if not pools:
        sys.exit("no complete model available for this scheme")

    # ---- long summary (all models, fractions, raw/cal)
    long = []
    for label, p in pools.items():
        for frac in fracs:
            for cal in (["raw", "cal"] if frac == 1.0 else ["cal"]):
                for r in summarize(p, co.tasks, frac, cal, a.n_boot if frac == 1.0 else 0):
                    long.append(dict(model=label, **r))
    long = pd.DataFrame(long)
    long.to_csv(mdir / "summary_long.csv", index=False)
    full = long[(long.frac == 1.0) & (long.cal == "cal")]

    def get(model, task, metric, df=full):
        r = df[(df.model == model) & (df.task == task) & (df.metric == metric)]
        return r.iloc[0].to_dict() if len(r) else {}

    # ---- Table 1: dataset
    cls = pd.read_csv(P["audit"] / "class_distribution.csv")
    split_sum = pd.read_csv(P["splits"] / "split_summary.csv")
    prim = split_sum[split_sum.split == "primary"].set_index("part")
    t1 = []
    for r in cls.itertuples():
        t1.append(dict(Task=TASK_NAMES[r.task], Class=r.class_value, Patients=r.patients_metadata, Images=r.images_metadata,
                       **{f"{k} (patients)": int(prim.loc[k, r.class_value]) if r.class_value in prim.columns else None
                          for k in ("train", "val", "test")}))
    save(pd.DataFrame(t1), "table1_dataset", tabs, "EGID patient and image counts per class and primary-split cohort.")

    # ---- Table 2: architectures & hyper-parameters
    tuned = cfg.get("tuned", {})
    t2 = []
    for label, v, e in models:
        vc = {**cfg["variants"], **cfg["finetune_variants"]}[v]
        sub = recs[(recs.variant == v) & (recs.encoder == e)]
        hp = {**cfg["train"], **(tuned.get(vc["kind"], {}) if not vc.get("finetune") else {}), **vc}
        t2.append(dict(Model=label, Description=vc["desc"], Encoder=e, Pooling=vc.get("pooling", "mean of image probs"),
                       Head=vc.get("head", "independent"), ViewDropout=vc.get("view_dropout", False),
                       Consistency=vc.get("consistency", False), Loss=hp.get("supervised_loss", "ce"),
                       lr=hp["lr"], wd=hp["weight_decay"], hidden=hp["hidden"], dropout=hp["dropout"],
                       input_dropout=hp.get("input_dropout", 0.0),
                       TrainableParams=int(sub.n_params.median()) if len(sub) else None))
    save(pd.DataFrame(t2), "table2_models", tabs, "Model variants and hyper-parameters.")

    # ---- Table 3: main results, four tasks
    t3 = []
    for label in pools:
        row = dict(Model=label)
        for t in ["hp", "atrophy", "distribution", "type"]:
            row[f"{TASK_NAMES[t]} macro-F1"] = fmt(get(label, t, "macro_f1"))
            row[f"{TASK_NAMES[t]} bal.acc"] = fmt(get(label, t, "balanced_accuracy"))
        row["Mean macro-F1"] = fmt(get(label, "mean", "macro_f1"))
        row["Mean accuracy"] = fmt(get(label, "mean", "accuracy"))
        row["Mean AUROC"] = fmt(get(label, "mean", "auroc"))
        row["Coherent predictions"] = fmt(get(label, "mean", "coherence"))
        t3.append(row)
    save(pd.DataFrame(t3), "table3_main", tabs, f"Patient-level results ({a.scheme}); mean over seeds, 95\\% patient-bootstrap CI.")

    # ---- Table 4: distribution per-class
    t4 = []
    for label, p in pools.items():
        pc = per_class(p, co.tasks)
        for r in pc.itertuples():
            t4.append(dict(Model=label, Class=r.cls, Support=r.support, F1=round(r.f1, 3), F1_seed_sd=round(r.f1_sd, 3)
                           if not pd.isna(r.f1_sd) else None, Recall=round(r.recall, 3), Precision=round(r.precision, 3)))
    save(pd.DataFrame(t4), "table4_distribution_per_class", tabs, "Atrophy-distribution per-class metrics.")
    conf = {label: confusion(p).tolist() for label, p in pools.items()}
    (mdir / "distribution_confusion.json").write_text(json.dumps(conf))

    # ---- statistics: paired comparisons
    comps = []
    ref = "A6" if "A6" in pools else None
    pairs = [("A6", m) for m in pools if m != "A6"] + [("A1", "A0"), ("A1", "B2 IMG (efficientnet_b0)"),
                                                        ("A2", "A1"), ("A3", "A1"), ("A4", "A3"), ("A5", "A2"),
                                                        ("A6", "A5"), ("A0", "B2 IMG (efficientnet_b0)"),
                                                        ("M2", "A0"), ("M3", "A0"), ("M5", "M2"), ("M6", "M5"), ("M6", "A0"),
                                                        ("M6", "B2 IMG (efficientnet_b0)"), ("M6", "B3 IMG (densenet121)"),
                                                        # Addendum B (pre-registered)
                                                        *[(n, r) for n in ("N1", "N3", "N5") for r in ("B3 IMG (densenet121)", "A0", "B2 IMG (efficientnet_b0)")],
                                                        ("N3", "A1"), ("N4", "A6"), ("N2", "N1"), ("N6", "N5"),
                                                        # Addendum A (pre-registered): fine-tuned vs frozen, same encoder
                                                        *[(f"FT2_IMG ({e})", f"{b} IMG ({e})") for b, e in BASELINE_ENCODERS.items()],
                                                        *[(f"FT2_IMG ({e})", "A6") for e in BASELINE_ENCODERS.values()],
                                                        ("FT_A6 (efficientnet_b0)", "B3 IMG (densenet121)"),
                                                        # Addendum C (pre-registered)
                                                        ("C3", "A6"), ("C1", "A6"), ("C2", "A6"),
                                                        ("C3", "B2 IMG (efficientnet_b0)"), ("C3", "B3 IMG (densenet121)")]
    for ma, mb in dict.fromkeys(pairs):
        if ma in pools and mb in pools:
            for t, metric in [("distribution", "macro_f1"), ("distribution", "balanced_accuracy"), ("hp", "macro_f1")]:
                comps.append(dict(A=ma, B=mb, **paired(pools[ma], pools[mb], t, metric, 1.0, a.n_boot)))
            for frac in fracs[1:]:
                comps.append(dict(A=ma, B=mb, **paired(pools[ma], pools[mb], "distribution", "macro_f1", frac, a.n_boot)))
    comps = pd.DataFrame(comps)
    comps.to_csv(mdir / "paired_comparisons.csv", index=False)

    # ---- Table 5: ablations
    abl = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "X1", "X2", "X3", "X4", "M2", "M3", "M5", "M6"]
    comp_cfg = dict(A0="mean pooling", A1="task attention", A2="+ structure", A3="+ view dropout",
                    A4="+ view dropout + consistency", A5="+ structure + view dropout", A6="full CoRe-MIL",
                    X1="task attention + soft clinical loss", X2="CoRe-MIL, weighted CE", X3="CoRe-MIL, focal",
                    X4="CoRe-MIL, balanced sampling", M2="mean pooling + structure (post hoc)",
                    M3="mean pooling + view dropout (post hoc)", M5="mean + structure + view dropout (post hoc)",
                    M6="CoRe-MIL with mean pooling (post hoc)")
    ref_of = dict(A1="A0", A2="A1", A3="A1", A4="A3", A5="A2", A6="A5", X1="A1", X2="A6", X3="A6", X4="A6",
                  M2="A0", M3="A0", M5="M2", M6="M5")
    t5 = []
    for m in abl:
        if m not in pools:
            continue
        row = dict(Exp=m, Components=comp_cfg[m], **{"Dist macro-F1": fmt(get(m, "distribution", "macro_f1")),
                                                     "Dist bal.acc": fmt(get(m, "distribution", "balanced_accuracy")),
                                                     "Mean macro-F1": fmt(get(m, "mean", "macro_f1")),
                                                     "Dist ECE": fmt(get(m, "distribution", "ece")),
                                                     "Coherence": fmt(get(m, "mean", "coherence")),
                                                     "Dist macro-F1 @25%": fmt(get(m, "distribution", "macro_f1",
                                                                                   long[(long.frac == 0.25)]))})
        rm = ref_of.get(m)
        c = comps[(comps.A == m) & (comps.B == rm) & (comps.task == "distribution") & (comps.metric == "macro_f1") &
                  (comps.frac == 1.0)] if len(comps) else []
        if len(c):
            c = c.iloc[0]
            row[f"Δ vs ref"] = f"{rm}: {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}], p={c['p_boot']:.2f}"
        t5.append(row)
    save(pd.DataFrame(t5), "table5_ablation", tabs, "Ablation study; Δ = paired patient-bootstrap difference in "
                                                    "atrophy-distribution macro-F1 versus the reference variant.")

    # ---- Table 6: missing-view robustness
    t6 = []
    for label in pools:
        for frac in fracs:
            sub = long[(long.model == label) & (long.frac == frac) & (long.cal == "cal")]
            g = lambda t, m: sub[(sub.task == t) & (sub.metric == m)].iloc[0] if len(sub[(sub.task == t) & (sub.metric == m)]) else None  # noqa
            r1, r2, r3 = g("distribution", "macro_f1"), g("distribution", "balanced_accuracy"), g("distribution", "ece")
            if r1 is None:
                continue
            t6.append(dict(Model=label, Fraction=frac, DistMacroF1=round(r1.estimate, 3), RepSD=round(r1.rep_sd, 3),
                           SeedSD=round(r1.seed_sd, 3), DistBalAcc=round(r2.estimate, 3), DistECE=round(r3.estimate, 3),
                           MeanMacroF1=round(g("mean", "macro_f1").estimate, 3), Reps=int(r1.n_reps)))
    t6 = pd.DataFrame(t6)
    if len(t6):
        base = t6[t6.Fraction == 1.0].set_index("Model").DistMacroF1
        t6["DropFromFull"] = (t6.DistMacroF1 - t6.Model.map(base)).round(3)
    save(t6, "table6_missing_view", tabs, "Random image-subset robustness (atrophy distribution). RepSD: between "
                                          "subset-draw SD; SeedSD: between training-seed SD.")

    # ---- Table 7: calibration + key statistical comparisons
    t7 = []
    for label in pools:
        raw = long[(long.model == label) & (long.frac == 1.0) & (long.cal == "raw")]
        for t in ["distribution", "hp"]:
            gr = lambda df, m: df[(df.task == t) & (df.metric == m)].estimate.iloc[0]  # noqa
            calr = full[full.model == label]
            t7.append(dict(Model=label, Task=TASK_NAMES[t], ECE_raw=round(gr(raw, "ece"), 3), ECE_cal=round(gr(calr, "ece"), 3),
                           NLL_raw=round(gr(raw, "nll"), 3), NLL_cal=round(gr(calr, "nll"), 3),
                           Brier_raw=round(gr(raw, "brier"), 3), Brier_cal=round(gr(calr, "brier"), 3)))
    save(pd.DataFrame(t7), "table7a_calibration", tabs, "Calibration before (raw) and after validation-fitted temperature scaling.")
    if len(comps):
        t7b = comps[(comps.frac == 1.0)][["A", "B", "task", "metric", "diff", "lo", "hi", "p_boot", "std_effect",
                                          "mcnemar_a_only", "mcnemar_b_only", "mcnemar_p", "n_patients", "n_seeds"]].round(4)
        save(t7b, "table7b_comparisons", tabs, "Paired patient-bootstrap comparisons (A minus B) and exact McNemar tests "
                                               "on seed-ensembled predictions. p-values are unadjusted.")
    rel = []
    for label, p in pools.items():
        for cal in ("raw", "cal"):
            rt = reliability_table(p, "distribution", cal)
            rt["model"], rt["cal"] = label, cal
            rel.append(rt)
    pd.concat(rel).to_csv(mdir / "reliability_distribution.csv", index=False)

    # ---- Table 8: compute
    t8 = []
    feat_meta = {p.stem: json.loads(p.read_text()) for p in (P["cache"] / "features").glob("*.json")}
    for label, v, e in models:
        sub = recs[(recs.variant == v) & (recs.encoder == e) & recs.split.isin(["cv0", "cv1", "cv2", "cv3", "cv4", "primary"])]
        if not len(sub):
            continue
        fm = feat_meta.get(f"{e}_224", {})
        t8.append(dict(Model=label, Runs=len(sub), TrainableParams=int(sub.n_params.median()),
                       EncoderParams=fm.get("n_params"), FeatureExtraction_s=round(fm.get("seconds", np.nan), 1),
                       TrainTime_s_median=round(sub.train_seconds.median(), 1), Epochs_median=int(sub.epochs_run.median()),
                       BestEpoch_median=int(sub.best_epoch.median()),
                       Device="mps (train) / cpu (eval)" if v in cfg["finetune_variants"] else sub.device.iloc[0]))
    save(pd.DataFrame(t8), "table8_compute", tabs, "Compute cost on Apple M1 (16 GB). Feature extraction on MPS; "
                                                   "cached-feature training on CPU.")

    # ---- supplementary: encoder sweep
    enc_rows = []
    for v in ["IMG", "A0", "A1", "A6"]:
        for e in cfg["encoders"]:
            p = Pooled(recs, pred_dir, v, e, 224, a.scheme)
            if p.ok:
                rows = summarize(p, co.tasks, 1.0, "cal", a.n_boot)
                d = {(r["task"], r["metric"]): r for r in rows}
                enc_rows.append(dict(Variant=v, Encoder=e, Seeds=len(p.seeds),
                                     DistMacroF1=fmt(d[("distribution", "macro_f1")]),
                                     MeanMacroF1=fmt(d[("mean", "macro_f1")]), HpMacroF1=fmt(d[("hp", "macro_f1")])))
    if enc_rows:
        save(pd.DataFrame(enc_rows), "tableS1_encoders", tabs, "Encoder sweep (frozen features).")
    print("tables ->", tabs)
