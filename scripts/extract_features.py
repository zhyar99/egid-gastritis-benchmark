"""Phase 3a: cache frozen embeddings for each encoder."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import get_device, load_config, setup_logging  # noqa: E402
from src.data.dataset import Cohort  # noqa: E402
from src.data.preprocessing import open_cache  # noqa: E402
from src.training.feature_extraction import extract, extract_augmented  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    fe = cfg["feature_extraction"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="*", default=list(cfg["encoders"]))
    ap.add_argument("--size", type=int, default=fe["image_size"])
    ap.add_argument("--cache-size", type=int, default=None, help="image cache to read (default: --size)")
    ap.add_argument("--batch-size", type=int, default=fe["batch_size"])
    ap.add_argument("--device", default=cfg["device"])
    ap.add_argument("--aug-views", type=int, default=0, help="also cache N augmented copies per image (Addendum C)")
    a = ap.parse_args()
    setup_logging("extract_features")
    P = cfg["paths"]
    co = Cohort.load(P["data"], P["audit"])
    images = open_cache(P["cache"], a.cache_size or a.size, co.index)
    dev = get_device(a.device)
    for enc in a.encoders:
        extract(images, co.index.image_id.tolist(), enc, cfg["encoders"], a.size, P["cache"], dev,
                a.batch_size, fe["hflip_tta"])
        if a.aug_views:
            extract_augmented(images, co.index.image_id.tolist(), enc, cfg["encoders"], a.size, P["cache"], dev,
                              a.aug_views, a.batch_size)
