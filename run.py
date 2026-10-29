"""Runs the experiments in order.

    python run.py                 # steps 1 to 8
    python run.py --from 3        # from step 3 on
    python run.py --from 2 --to 4
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = sorted(p.name for p in HERE.glob("[0-9][0-9]_*.py"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="first", type=int, default=1)
    ap.add_argument("--to", dest="last", type=int, default=len(STEPS))
    args = ap.parse_args()
    for step in STEPS:
        if not args.first <= int(step[:2]) <= args.last:
            continue
        print(f"\n===== {step} =====", flush=True)
        r = subprocess.run([sys.executable, str(HERE / step)], cwd=HERE)
        if r.returncode != 0:
            sys.exit(f"Step {step} ended with code {r.returncode}.")


if __name__ == "__main__":
    main()
