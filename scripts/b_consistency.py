"""Consistency of two groups of runs of the same configuration (docs/lab-notebook.md, 2026-10-14, phase 3).

Reads eval_<split>.json of each run; per metric: mean and sample std (ddof=1) per group, difference of means
(b - a), Welch standard error sqrt(s_a^2/n_a + s_b^2/n_b) and the ratio difference / SE. Ratios with
|ratio| > --flag (default 3) are flagged. Validation only by default; values in percent.

Usage:
    python scripts/b_consistency.py --a ~/runs/night1/ep133_s{0,1,2} --b ~/runs/night3/B_s{3,4,5,6,7}
"""

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ["rel_all", "rel_p", "rel_vx", "rel_vy"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", nargs="+", required=True, type=Path)
    ap.add_argument("--b", nargs="+", required=True, type=Path)
    ap.add_argument("--split", default="val", choices=["val"])
    ap.add_argument("--flag", type=float, default=3.0)
    args = ap.parse_args()
    groups = {}
    for name, runs in (("a", args.a), ("b", args.b)):
        evs = [json.loads((r / f"eval_{args.split}.json").read_text())["means"] for r in runs]
        groups[name] = {m: 100 * np.array([e[m] for e in evs]) for m in METRICS}
        print(f"group {name} (n={len(runs)}): " + ", ".join(f"{r.parent.name}/{r.name}" for r in runs))
    print(f"{'metric':8s} {'mean a':>8s} {'sd a':>7s} {'mean b':>8s} {'sd b':>7s} {'b - a':>8s} {'SE':>7s} "
          f"{'diff/SE':>8s}")
    for m in METRICS:
        a, b = groups["a"][m], groups["b"][m]
        se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
        d = b.mean() - a.mean()
        flag = "  <-- |diff/SE| > %g" % args.flag if abs(d / se) > args.flag else ""
        print(f"{m:8s} {a.mean():8.4f} {a.std(ddof=1):7.4f} {b.mean():8.4f} {b.std(ddof=1):7.4f} {d:+8.4f} "
              f"{se:7.4f} {d / se:+8.2f}{flag}")
        print(f"{'':8s} a: {' '.join(f'{x:.4f}' for x in a)} | b: {' '.join(f'{x:.4f}' for x in b)}")


if __name__ == "__main__":
    main()
