"""Approximate seen- vs unseen-shape table from existing results (EXPLORATORY; docs/lab-notebook.md 2026-10-15).

CPU only, no run and no inference: reads the per-run CSVs written by scripts/train_fit.py (final model in eval mode):
  train_rel_*  on the training part of the grouped split: seen shapes AND seen samples (used in training)
  val_rel_*    on the grouped validation: unseen shapes
For every configuration, rel_all and rel_vy on both, as the difference to B in Welch standard errors
(diff / SE = (mean - mean_B) / sqrt(s^2/n + s_B^2/n_B), std ddof=1, all seeds available).
Limit: the training set is not a true random validation (its samples were fitted); the night-6 grouped-holdout
experiment is the clean version.

Usage:
    python scripts/seen_proxy.py --runs-csv docs/results/train-fit/train_fit_runs.csv \
        --runs-csv docs/results/train-fit-positions-all/train_fit_runs.csv \
        --out-dir docs/results/seen-proxy --fig-dir docs/figures/seen-proxy
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_val as av
from train_fit import welch

CONFIGS = ["MD8", "SA8", "CBAM8", "B-large", "MD32", "SA32", "MDv8", "MDv32", "MDv64", "MDv8-16-32"]
METRICS = ("rel_all", "rel_vy")
COLOR = {"train": "#D55E00", "val": "#0072B2"}  # validated pair (positions figure, 2026-10-15)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-csv", type=Path, action="append", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    runs = {}  # (config, seed) -> row; the same run in two files must agree
    for path in args.runs_csv:
        for r in csv.DictReader(open(path)):
            key = (r["config"], int(r["seed"]))
            vals = {f"{s}_{m}": float(r[f"{s}_{m}"]) for s in ("train", "val") for m in METRICS}
            if key in runs:
                diff = max(abs(runs[key][k] - vals[k]) for k in vals)
                assert diff < 1e-5, f"{key}: files disagree by {diff}"
            runs[key] = vals
    data = {c: {k: np.array([v[k] for (cc, _), v in sorted(runs.items()) if cc == c])
                for k in ("train_rel_all", "train_rel_vy", "val_rel_all", "val_rel_vy")} for c in ["B", *CONFIGS]}
    configs = [c for c in CONFIGS if data[c]["train_rel_all"].size]

    rows, z = [], {}
    for c in configs:
        n = data[c]["train_rel_all"].size
        row = {"config": c, "seeds": f"{n}/{data['B']['train_rel_all'].size}"}
        for m in METRICS:
            for s, name in (("train", "train (seen shapes, seen samples)"), ("val", "val (unseen shapes)")):
                d, se, zz = welch(data[c][f"{s}_{m}"], data["B"][f"{s}_{m}"])
                z[(c, s, m)] = zz
                row[f"{m} {name}: diff % (diff/SE)"] = f"{100 * d:+.4f} ({zz:+.2f})"
            row[f"{m} z_train - z_val"] = f"{z[(c, 'train', m)] - z[(c, 'val', m)]:+.2f}"
        rows.append(row)
    base = {m: {s: f"{100 * data['B'][f'{s}_{m}'].mean():.4f}" for s in ("train", "val")} for m in METRICS}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    av.write_csv(args.out_dir / "seen_proxy.csv", rows)
    md = ["# Approximate seen- vs unseen-shape comparison from train_fit results (EXPLORATORY)", "",
          "Sources: " + ", ".join(f"`{p.as_posix()}`" for p in args.runs_csv) + ". Final models in eval mode; train = "
          "training part of the grouped split (5998 samples: seen shapes and seen samples), val = grouped validation "
          "(402 samples, unseen shapes). diff = mean - mean_B over seeds (percentage points); SE = Welch. "
          f"B means: rel_all train {base['rel_all']['train']} / val {base['rel_all']['val']} %, rel_vy train "
          f"{base['rel_vy']['train']} / val {base['rel_vy']['val']} %. **Limit:** the training set is not a true "
          "random validation (its samples were fitted); the night-6 grouped-holdout experiment is the clean version. "
          "3-seed configurations: rough.", "", av.md_table(rows, list(rows[0].keys())), ""]
    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # --- figure: IEEE single column, diff/SE vs B on train (seen) and val (unseen), one panel per metric;
    # grey band = |diff/SE| < 2; the two markers of a configuration are offset vertically so neither hides ---
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "legend.fontsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.6), sharey=True, constrained_layout=True)
    y = np.arange(len(configs))[::-1]
    for ax, m, title in zip(axes, METRICS, ("(a) rel_all", "(b) rel_vy")):
        ax.axvspan(-2, 2, color="#f2f2f2", zorder=0, lw=0)
        ax.axvline(0, color="#999999", lw=0.6, zorder=1)
        for yi, c in zip(y, configs):
            zt, zv = z[(c, "train", m)], z[(c, "val", m)]
            ax.plot([zt, zv], [yi + 0.13, yi - 0.13], color="#cccccc", lw=0.8, zorder=2)
            ax.scatter([zt], [yi + 0.13], s=16, marker="o", color=COLOR["train"], edgecolors="white", linewidths=0.5,
                       zorder=3, label="train: seen shapes + samples" if yi == y[0] else None)
            ax.scatter([zv], [yi - 0.13], s=16, marker="D", facecolors="white", edgecolors=COLOR["val"], linewidths=1.0,
                       zorder=3, label="val: unseen shapes" if yi == y[0] else None)
        ax.set_title(title, loc="left")
        ax.set_xlabel("difference to B (Welch SE)")
        ax.grid(axis="x", color="#eeeeee", lw=0.5)
    axes[0].set_yticks(y, [f"{c} (n=3)" if data[c]["train_rel_all"].size == 3 else c for c in configs])
    axes[0].set_ylim(-0.7, len(configs) - 0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, handletextpad=0.3,
               columnspacing=1.0)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig_dir / "seen_proxy.pdf")
    fig.savefig(args.fig_dir / "seen_proxy.png", dpi=300)
    plt.close(fig)
    print(f"Wrote {args.fig_dir}/seen_proxy.pdf (+ .png preview)")


if __name__ == "__main__":
    main()
