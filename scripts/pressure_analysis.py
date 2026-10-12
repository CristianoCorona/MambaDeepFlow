"""Per-sample analysis of the pressure error (rel_p) from the evaluation CSVs; no training, no new evaluation.

For each group of runs with the same budget and validation split (seeds as members):
  - distribution of per-sample rel_p (quantiles) per run
  - the 10 samples that weigh most on the mean rel_p (seed-averaged rel_p; share of the sum)
  - relation between rel_p and sum|p target| of the sample (Spearman; mean rel_p and sum|o-t| per quintile)
  - seed consistency on the same sample (Spearman between seeds, per-sample spread)
  - whether the seed-to-seed difference of the mean rel_p is dominated by a few samples
Reads eval_val_samples.csv (with the abs_err_p / abs_tgt_p columns of scripts/evaluate.py, 2026-10-12).
Writes docs/results/pressure_top10.csv. Validation only; the test set is not read.

Usage:
    python scripts/pressure_analysis.py [--night ~/runs/night1] [--calib ~/runs/calib]
"""

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
QUANTILES = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
TOP = 10


def read_samples(run_dir):
    with open(run_dir / "eval_val_samples.csv") as fh:
        rows = list(csv.DictReader(fh))
    return {"file": [r["file"] for r in rows],
            "rel_p": np.array([float(r["rel_p"]) for r in rows]),
            "err_p": np.array([float(r["abs_err_p"]) for r in rows]),
            "tgt_p": np.array([float(r["abs_tgt_p"]) for r in rows])}


def ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def profile(name):
    return name[:-len(".npz")].rsplit("_", 2)[0]


def analyse(label, run_dirs, top_rows):
    runs = {d.name: read_samples(d) for d in run_dirs}
    names = list(runs)
    files = runs[names[0]]["file"]
    assert all(runs[n]["file"] == files for n in names), "sample order differs between runs"
    tgt = runs[names[0]]["tgt_p"]
    assert all(np.allclose(runs[n]["tgt_p"], tgt, rtol=1e-9) for n in names)
    P = np.vstack([runs[n]["rel_p"] for n in names]) * 100  # (seeds, samples), %
    E = np.vstack([runs[n]["err_p"] for n in names])
    n = P.shape[1]
    print(f"\n=== {label}: runs {', '.join(names)}; {n} samples; rel_p in %")

    print("Per-sample rel_p quantiles (min / 10 / 25 / 50 / 75 / 90 / 95 / 99 / max), mean:")
    for name, row in zip(names, P):
        q = np.quantile(row, QUANTILES)
        print(f"  {name}: " + " / ".join(f"{v:.2f}" for v in q) + f"; mean {row.mean():.4f}")

    pm = P.mean(axis=0)
    order = np.argsort(-pm)
    share = pm[order[:TOP]].sum() / pm.sum()
    mean_wo = np.delete(P, order[:TOP], axis=1).mean(axis=1)
    print(f"Top {TOP} samples by seed-mean rel_p: {share:.1%} of the summed rel_p ({TOP / n:.1%} of samples); "
          f"mean rel_p without them: " + ", ".join(f"{v:.4f}" for v in mean_wo)
          + " (vs " + ", ".join(f"{v:.4f}" for v in P.mean(axis=1)) + ")")
    tgt_rank = ranks(tgt)  # 0 = smallest sum|p target|
    for i in order[:TOP]:
        print(f"  {files[i]:28s} rel_p {' / '.join(f'{v:.1f}' for v in P[:, i])}  sum|p t| {tgt[i]:8.2f} "
              f"(rank {int(tgt_rank[i]) + 1} of {n} from smallest)  sum|p o-t| {E[:, i].mean():.2f}")
        top_rows.append({"group": label, "file": files[i], "profile": profile(files[i]),
                         **{f"rel_p_{nm}": f"{v:.4f}" for nm, v in zip(names, P[:, i])},
                         "rel_p_seed_mean": f"{pm[i]:.4f}", "sum_abs_p_target": f"{tgt[i]:.6g}",
                         "sum_abs_p_err_seed_mean": f"{E[:, i].mean():.6g}"})

    print(f"rel_p vs sum|p target|: Spearman {spearman(pm, tgt):+.3f}; "
          f"sum|p o-t| vs sum|p target|: Spearman {spearman(E.mean(axis=0), tgt):+.3f}")
    edges = np.quantile(tgt, [0, 0.2, 0.4, 0.6, 0.8, 1])
    bins = np.clip(np.searchsorted(edges, tgt, side="right") - 1, 0, 4)
    print("  by quintile of sum|p target| (Q1 = smallest): mean rel_p / median sum|p o-t| / median sum|p t|")
    for b in range(5):
        m = bins == b
        print(f"  Q{b + 1}: {pm[m].mean():6.2f} / {np.median(E.mean(axis=0)[m]):7.2f} / {np.median(tgt[m]):8.2f}")

    print("Seed consistency (same sample):")
    for a, b in combinations(range(len(names)), 2):
        d = P[a] - P[b]
        ad = np.abs(d)
        idx = np.argsort(-ad)
        sq_share = (d[idx[:TOP]] ** 2).sum() / (d ** 2).sum()
        mean_diff_wo = np.delete(d, idx[:TOP]).mean()
        print(f"  {names[a]} vs {names[b]}: Spearman {spearman(P[a], P[b]):.3f}; median |diff| {np.median(ad):.2f}, "
              f"90th pct {np.quantile(ad, 0.9):.2f}, max {ad.max():.1f}; mean diff {d.mean():+.4f}; "
              f"top {TOP} |diff| samples carry {sq_share:.1%} of sum diff^2; "
              f"mean diff without them {mean_diff_wo:+.4f}; samples with diff > 0: {(d > 0).mean():.0%}, "
              f"median diff {np.median(d):+.3f}")
        print("    largest |diff|: " + ", ".join(f"{files[i]} ({d[i]:+.1f})" for i in idx[:5]))
    if len(names) >= 3:
        var = P.var(axis=0, ddof=1)
        idx = np.argsort(-var)
        print(f"  across {len(names)} seeds: top {TOP} samples carry {var[idx[:TOP]].sum() / var.sum():.1%} of the "
              f"summed per-sample variance; sd of run means {P.mean(axis=1).std(ddof=1):.4f}, without them "
              f"{np.delete(P, idx[:TOP], axis=1).mean(axis=1).std(ddof=1):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--night", type=Path, default=Path.home() / "runs" / "night1")
    ap.add_argument("--calib", type=Path, default=Path.home() / "runs" / "calib")
    args = ap.parse_args()
    top_rows = []
    analyse("calibration 133 ep, random val", [args.calib / f"ep133_s{s}" for s in (0, 1)], top_rows)
    analyse("night1 133 ep, grouped val", [args.night / f"ep133_s{s}" for s in (0, 1, 2)], top_rows)
    analyse("night1 67 ep, grouped val", [args.night / f"ep67_s{s}" for s in (0, 1)], top_rows)
    out = REPO / "docs" / "results" / "pressure_top10.csv"
    fields = sorted({k for r in top_rows for k in r}, key=lambda k: (not k in ("group", "file", "profile"), k))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(top_rows)
    print(f"\nWrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
