"""Independent recomputation of a two-group comparison from the per-sample evaluation CSVs.

Does not use scripts/compare_runs.py or eval_<split>.json for the numbers: per-run means are
recomputed from eval_<split>_samples.csv (and checked against the JSON means), then group mean and
standard deviation with both ddof=1 (sample std) and ddof=0 (population std), and the 6c criterion
|mean_A - mean_B| <= k * max(std_A, std_B) with each convention.

Usage:
    python scripts/recheck_groups.py --a RUN... --b RUN... [--split test] [--k 2]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = ["rel_p", "rel_vx", "rel_vy", "rel_v", "rel_all"]
CRITERION_METRICS = ["rel_p", "rel_v", "rel_all"]  # pre-registered for 6c


def run_means(run, split):
    with open(Path(run) / f"eval_{split}_samples.csv") as fh:
        rows = list(csv.DictReader(fh))
    means = {m: float(np.mean([float(r[m]) for r in rows])) for m in METRICS}
    ref = json.loads((Path(run) / f"eval_{split}.json").read_text())["means"]
    worst = max(abs(means[m] - ref[m]) for m in METRICS)
    return means, len(rows), worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--k", type=float, default=2.0)
    args = ap.parse_args()

    groups = {}
    for g, runs in (("A", args.a), ("B", args.b)):
        groups[g] = []
        for r in runs:
            means, n, worst = run_means(r, args.split)
            print(f"{g} {Path(r).name:10s} n={n}  max |CSV mean - JSON mean| = {worst:.2e}")
            groups[g].append(means)

    print("\nRelative errors in %. Columns: mean, std ddof=1, std ddof=0")
    print(f"{'metric':8s} {'mean A':>8s} {'sA1':>6s} {'sA0':>6s} {'mean B':>8s} {'sB1':>6s} {'sB0':>6s} "
          f"{'|d|':>6s} {'lim1':>6s} {'lim0':>6s} {'ddof1':>5s} {'ddof0':>5s}")
    verdict = {1: True, 0: True}
    for m in METRICS:
        xa = np.array([e[m] for e in groups["A"]]) * 100
        xb = np.array([e[m] for e in groups["B"]]) * 100
        d = abs(xa.mean() - xb.mean())
        lim = {dd: args.k * max(xa.std(ddof=dd), xb.std(ddof=dd)) for dd in (1, 0)}
        ok = {dd: d <= lim[dd] for dd in (1, 0)}
        crit = m in CRITERION_METRICS
        for dd in (1, 0):
            if crit:
                verdict[dd] &= ok[dd]
        tag = {dd: ("pass" if ok[dd] else "FAIL") if crit else "info" for dd in (1, 0)}
        print(f"{m:8s} {xa.mean():8.4f} {xa.std(ddof=1):6.4f} {xa.std(ddof=0):6.4f} "
              f"{xb.mean():8.4f} {xb.std(ddof=1):6.4f} {xb.std(ddof=0):6.4f} "
              f"{d:6.4f} {lim[1]:6.4f} {lim[0]:6.4f} {tag[1]:>5s} {tag[0]:>5s}")
    for dd in (1, 0):
        print(f"Criterion ({', '.join(CRITERION_METRICS)}, k={args.k:g}) with ddof={dd}: "
              f"{'compatible' if verdict[dd] else 'NOT compatible'}")


if __name__ == "__main__":
    main()
