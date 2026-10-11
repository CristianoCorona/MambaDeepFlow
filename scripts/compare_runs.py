"""Compare runs: loss curves of two runs (same seed), or evaluation summaries of two groups.

curves:  python scripts/compare_runs.py curves RUN_A RUN_B
         per-epoch differences of train/val L1 from metrics.csv (max abs, max rel, final values).
groups:  python scripts/compare_runs.py groups --a RUN... --b RUN... [--split test] [--k 2]
         mean and sample std (ddof=1) per group of eval_<split>.json means, difference of means,
         and the pre-registered compatibility criterion (docs/lab-notebook.md):
         |mean_a - mean_b| <= k * max(std_a, std_b) for rel_p, rel_v, rel_all.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

CRITERION_METRICS = ["rel_p", "rel_v", "rel_all"]
REPORT_METRICS = ["rel_p", "rel_vx", "rel_vy", "rel_v", "rel_all",
                  "fluid_rel_p", "fluid_rel_v", "fluid_rel_all", "L1", "L1_denorm"]


def read_metrics(run):
    with open(Path(run) / "metrics.csv") as fh:
        return list(csv.DictReader(fh))


def curves(a, b):
    ra, rb = read_metrics(a), read_metrics(b)
    if len(ra) != len(rb):
        sys.exit(f"Different number of epochs: {len(ra)} vs {len(rb)}")
    print(f"A: {a}\nB: {b}\nEpochs: {len(ra)}")
    for col in ("train_L1_mean_of_batches", "val_L1_mean_of_batches", "lr"):
        xa = np.array([float(r[col]) for r in ra])
        xb = np.array([float(r[col]) for r in rb])
        d = np.abs(xa - xb)
        rel = d / np.abs(xa)
        i = int(np.argmax(rel))
        print(f"{col}:\n  max|A-B| {d.max():.3e}  mean|A-B| {d.mean():.3e}  "
              f"max rel {rel.max():.3e} (epoch {i})  identical epochs {int(np.sum(d == 0))}/{len(d)}")
        print(f"  final A {xa[-1]:.8g}  B {xb[-1]:.8g}  rel diff {rel[-1]:.3e}")


def load_eval(run, split):
    return json.loads((Path(run) / f"eval_{split}.json").read_text())["means"]


def groups(a_runs, b_runs, split, k):
    if len(a_runs) < 2 or len(b_runs) < 2:
        sys.exit("Need at least 2 runs per group (sample std)")
    ea = [load_eval(r, split) for r in a_runs]
    eb = [load_eval(r, split) for r in b_runs]
    print(f"Group A (n={len(ea)}): " + ", ".join(Path(r).name for r in a_runs))
    print(f"Group B (n={len(eb)}): " + ", ".join(Path(r).name for r in b_runs))
    print("Relative errors in %; std = sample std (ddof=1)")
    print(f"{'metric':14s} {'mean A':>9s} {'std A':>8s} {'mean B':>9s} {'std B':>8s} "
          f"{'|dA-B|':>8s} {'k*maxstd':>9s}  result")
    all_ok = True
    for m in REPORT_METRICS:
        scale = 1.0 if m.startswith("L1") else 100.0
        xa = np.array([e[m] for e in ea]) * scale
        xb = np.array([e[m] for e in eb]) * scale
        sa, sb = xa.std(ddof=1), xb.std(ddof=1)
        diff = abs(xa.mean() - xb.mean())
        lim = k * max(sa, sb)
        if m in CRITERION_METRICS:
            ok = diff <= lim
            all_ok &= ok
            res = "compatible" if ok else "NOT compatible"
        else:
            res = "(info)"
        print(f"{m:14s} {xa.mean():9.4f} {sa:8.4f} {xb.mean():9.4f} {sb:8.4f} {diff:8.4f} {lim:9.4f}  {res}")
        print(f"{'':14s} A: {' '.join(f'{x:.4f}' for x in xa)} | B: {' '.join(f'{x:.4f}' for x in xb)}")
    print(f"Criterion ({', '.join(CRITERION_METRICS)}, k={k}): "
          f"{'all compatible' if all_ok else 'NOT all compatible'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curves")
    c.add_argument("run_a")
    c.add_argument("run_b")
    g = sub.add_parser("groups")
    g.add_argument("--a", nargs="+", required=True)
    g.add_argument("--b", nargs="+", required=True)
    g.add_argument("--split", default="test", choices=["val", "test"])
    g.add_argument("--k", type=float, default=2.0)
    args = ap.parse_args()
    if args.cmd == "curves":
        curves(args.run_a, args.run_b)
    else:
        groups(args.a, args.b, args.split, args.k)


if __name__ == "__main__":
    main()
