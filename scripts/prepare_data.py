"""Phase 2: patient-level splits and resized image caches (requires the audit outputs)."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_config, setup_logging  # noqa: E402
from src.data.metadata import load_metadata  # noqa: E402
from src.data.preprocessing import build_image_cache  # noqa: E402
from src.data.splits import make_all  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=cfg["preprocessing"]["sizes"])
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    setup_logging("prepare_data")
    P = cfg["paths"]
    meta = load_metadata(P["data"])
    groups = pd.read_csv(P["audit"] / "patient_groups.csv", dtype=str).set_index("patient_id").split_group.to_dict()
    make_all(meta, groups, P["splits"], seed=a.seed)
    print(pd.read_csv(P["splits"] / "split_summary.csv").to_string(index=False))
    index = pd.read_csv(P["audit"] / "image_index.csv", dtype={"patient_id": str})
    index = index[index.ok].reset_index(drop=True)
    for s in a.sizes:
        print("cache:", build_image_cache(index, P["data"], P["cache"], s, cfg["preprocessing"]["workers"]))
