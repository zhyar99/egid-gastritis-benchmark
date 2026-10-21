"""Phase 1: audit the EGID data folder. Read-only on data/."""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.audit import run_audit  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/data_audit")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    r = run_audit(Path(a.data), Path(a.out), a.workers)
    print(r["summary"].to_string(index=False))
    print("problems:", len(r["problems"]), "| cross-patient visual dups:",
          int((r["dups"].cross_patient & r["dups"].visual_duplicate).sum()), f"| {time.time() - t0:.0f}s")
