"""Summary figure "where and how much Mamba" for the position experiment (notebook 2026-10-15; validation only,
EXPLORATORY), plus the train / validation gap vs B in Welch SE (verification 5a style).

CPU only: reads the per-run CSV written by scripts/train_fit.py (train L1 in eval mode, val L1, val rel_all per seed).
  left panel   validation rel_all per configuration: one dot per seed, a dash for the mean
  right panel  train L1 (eval mode) vs validation L1: small dots per seed, a large marker for the mean
Configurations with 3 seeds (night 5) are drawn with hollow markers and marked "(n=3)".
Colours: grey = B, vermillion = MD (simplified Mamba block), blue = MDv (faithful SS2D); markers encode the position,
so identity does not rely on colour alone.

Usage:
    python scripts/positions_figure.py --runs-csv docs/results/train-fit-positions-all/train_fit_runs.csv \
        --fig-dir docs/figures/positions --out-dir docs/results/positions
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

# x order: by position, then number of blocks
CONFIGS = ["B", "MD8", "MDv8", "MD32", "MDv32", "MDv64", "MDv8-16-32"]
GROUPS = [("none", ["B"]), ("8x8", ["MD8", "MDv8"]), ("32x32", ["MD32", "MDv32"]), ("64x64", ["MDv64"]),
          ("8+16+32", ["MDv8-16-32"])]
COLOR = {"B": "#555555", "MD": "#D55E00", "MDv": "#0072B2"}
MARKER = {"B": "o", "8": "s", "32": "D", "64": "^", "8-16-32": "P"}


def family(c):
    return "B" if c == "B" else ("MDv" if c.startswith("MDv") else "MD")


def position(c):
    return "B" if c == "B" else c[len(family(c)):]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-csv", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    rows = list(csv.DictReader(open(args.runs_csv)))
    data = {c: {k: np.array([float(r[k]) for r in rows if r["config"] == c])
                for k in ("seed", "train_L1", "val_L1", "gap_L1", "val_rel_all")} for c in CONFIGS}
    missing = [c for c in CONFIGS if data[c]["seed"].size == 0]
    if missing:
        sys.exit(f"no runs for {missing} in {args.runs_csv}")
    nseed = {c: data[c]["seed"].size for c in CONFIGS}

    # --- gap / train / val vs B, Welch SE, against B with the same seeds 0-2 and with all B seeds ---
    out = []
    for c in CONFIGS[1:]:
        for name, max_seed in (("seeds 0-2 (both)", 2), ("all seeds", 99)):
            sv, sb = data[c]["seed"] <= max_seed, data["B"]["seed"] <= max_seed
            row = {"config": c, "seeds": f"{int(sv.sum())}/{int(sb.sum())}", "subset": name}
            for k in ("train_L1", "val_L1", "gap_L1"):
                d, se, z = welch(data[c][k][sv], data["B"][k][sb])
                row[f"{k} - B x1e-3 (diff/SE)"] = f"{1e3 * d:+.4f} ({z:+.2f})"
            out.append(row)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    av.write_csv(args.out_dir / "gap_vs_B.csv", out)
    (args.out_dir / "gap_vs_B.md").write_text(
        "# Train L1 (eval), val L1 and gap vs B, Welch SE (position experiment, validation, EXPLORATORY)\n\n"
        f"Source: `{args.runs_csv.as_posix()}`. diff/SE = (mean - mean_B) / sqrt(s^2/n + s_B^2/n_B), std ddof=1. "
        "3 seeds: rough.\n\n" + av.md_table(out, list(out[0].keys())) + "\n")
    print(av.md_table(out, list(out[0].keys())))

    # --- figure ---
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "legend.fontsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.55), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    rng = np.random.default_rng(0)  # horizontal jitter of the seed dots only
    for j, c in enumerate(CONFIGS):
        col, mk, hollow = COLOR[family(c)], MARKER[position(c)], nseed[c] == 3
        v = 100 * data[c]["val_rel_all"]
        ax1.scatter(j + rng.uniform(-0.12, 0.12, v.size), v, s=9, marker=mk, linewidths=0.7,
                    facecolors="none" if hollow else col, edgecolors=col, alpha=0.85, zorder=2)
        ax1.plot([j - 0.3, j + 0.3], [v.mean()] * 2, color="#1a1a1a", lw=1.4, zorder=3, solid_capstyle="butt")
        tl, vl = 1e3 * data[c]["train_L1"], 1e3 * data[c]["val_L1"]
        ax2.scatter(tl, vl, s=6, marker=mk, linewidths=0.5, facecolors="none" if hollow else col, edgecolors=col,
                    alpha=0.45, zorder=2)
        ax2.scatter([tl.mean()], [vl.mean()], s=34, marker=mk, linewidths=1.0,
                    facecolors="white" if hollow else col, edgecolors=col, zorder=4)
    b = 100 * data["B"]["val_rel_all"].mean()
    ax1.axhline(b, color="#999999", lw=0.6, ls=(0, (3, 2)), zorder=1)
    edges = np.cumsum([len(m) for _, m in GROUPS])[:-1] - 0.5
    for e in edges:
        ax1.axvline(e, color="#dddddd", lw=0.6, zorder=0)
    ax1.set_xticks(range(len(CONFIGS)),
                   [f"{c}\n(n=3)" if nseed[c] == 3 else c for c in CONFIGS], rotation=90, fontsize=5.5)
    ax1.set_xlim(-0.6, len(CONFIGS) - 0.4)
    ax1.set_ylabel("validation rel_all (%)")
    ax1.set_title("(a) error per seed, mean (dash)", loc="left")
    ax1.grid(axis="y", color="#eeeeee", lw=0.5)

    # direct labels at the means; offsets in points, hand-placed to avoid overlap
    offs = {"B": (6, 0, "left"), "MD8": (-6, 0, "right"), "MDv8": (6, 2, "left"), "MD32": (6, 2, "left"),
            "MDv32": (0, 8, "center"), "MDv64": (5, -7, "left"), "MDv8-16-32": (-4, -8, "left")}
    for c in CONFIGS:
        tl, vl = 1e3 * data[c]["train_L1"].mean(), 1e3 * data[c]["val_L1"].mean()
        dx, dy, ha = offs[c]
        ax2.annotate(c, (tl, vl), xytext=(dx, dy), textcoords="offset points", fontsize=5.5, color="#1a1a1a",
                     ha=ha, va="center", zorder=5)
    ax2.set_xlabel("train L1, eval mode (x1e-3)")
    ax2.set_ylabel("validation L1 (x1e-3)")
    ax2.set_title("(b) fit vs generalisation", loc="left")
    ax2.grid(color="#eeeeee", lw=0.5)
    ax2.margins(x=0.12, y=0.08)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig_dir / "positions_summary.pdf")
    fig.savefig(args.fig_dir / "positions_summary.png", dpi=300)
    plt.close(fig)
    print(f"Wrote {args.fig_dir}/positions_summary.pdf (+ .png preview)")


if __name__ == "__main__":
    main()
