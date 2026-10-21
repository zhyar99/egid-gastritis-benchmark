"""Live view of training: final-model queue progress, the epochs of the model training now, and recent CV runs.

  python scripts/watch_training.py            # refreshes every 3 s; Ctrl+C to quit
"""
import collections
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
EXPORTED = {("A6", "efficientnet_b0"), ("IMG", "densenet121"), ("FT_A6", "efficientnet_b0")}  # kept in models/


def tail(p: Path, n: int = 4000) -> list[str]:
    try:
        return p.read_text(errors="ignore").splitlines()[-n:]
    except FileNotFoundError:
        return []


def planned() -> int:
    """Final models still to export: every (variant, encoder) with >= 5 CV runs, minus the ones in models/."""
    c = collections.Counter(tuple(re.match(r"(.+?)__(.+?)_224__", p.name).groups())
                            for p in (ROOT / "outputs" / "runs").glob("*__cv*__s*.json"))
    return sum(1 for k, n in c.items() if n >= 5 and k not in EXPORTED)


def screen() -> str:
    done = sorted((ROOT / "models" / "final").glob("*.pt"), key=lambda p: p.stat().st_mtime)
    out = [f"Final models (all 229 patients)  {time.strftime('%H:%M:%S')}",
           f"  exported: {len(done)} / {planned()}" + (f"   last: {done[-1].stem}" if done else "")]
    for name in ("final_all_frozen.out", "final_all_ft.out"):
        lines = tail(LOGS / name, 200)
        if not lines:
            continue
        state = "finished" if lines[-1].endswith("_DONE") else ("training " + lines[-1][4:] if lines[-1].startswith("===") else "between models")
        out.append(f"  {name[10:-4]:7s} queue: {state}")
    ep = [l for l in tail(LOGS / "train_final.log") if " train_final: " in l]
    out += ["", "Current model, epoch by epoch (training loss; no validation set in final training):"]
    out += ["  " + l.split(" train_final: ")[1] for l in ep[-12:]] or ["  (no epoch lines yet)"]
    cv = [l for l in tail(LOGS / "run_full.log") if "best epoch" in l]
    if cv:
        out += ["", "Last cross-validation runs (held-out validation results):"]
        out += ["  " + l.split("trainer: ")[1][:110] for l in cv[-5:]]
    return "\n".join(out)


if __name__ == "__main__":
    once = "--once" in sys.argv
    try:
        while True:
            s = screen()
            print(s if once else "\033[2J\033[H" + s + "\n\nCtrl+C to quit", flush=True)
            if once:
                break
            time.sleep(3)
    except KeyboardInterrupt:
        pass
