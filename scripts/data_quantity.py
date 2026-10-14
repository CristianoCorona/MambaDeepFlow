"""Data-quantity experiment: pre-registered analysis (docs/lab-notebook.md, 2026-10-14, commit 26b6e7f section 1, and
"Data-quantity experiment: implementation and queue"). EXPLORATORY, validation only (the test set is never read).

Groups: training fraction (realised share of the 5998 training files) x configuration (B, MD8, SA8), seeds 0-2.
The 100 % group is the main 133-epoch runs with the same seeds. Per run:
  rel_all, rel_p, rel_vy      validation, per-sample ratios (explore.cached_inference, batch 1, as evaluate.py)
  val_L1_final                last row of metrics.csv (training-time validation, mean of batches)
  train_L1 (eval)             final model in eval mode on the training files THIS run used (subset rebuilt with
                              mdata.subsample_per_profile; its sha256 must equal config.json), as scripts/train_fit.py
  val_L1 (eval), gap          validation L1 as train_fit.py; gap = val_L1 - train_L1
Per fraction: mean (sd) per configuration; MD8 - B and SA8 - B with the two-level bootstrap (seeds + profiles) on
rel_all / rel_p / rel_vy, Welch diff / SE for train L1, val L1 and gap. 3 seeds per group: rough.
Diagnostics only: validation-L1 curves (metrics.csv) and the position of their minimum (the main number is always the
final checkpoint).

Usage:
    python scripts/data_quantity.py --out-dir DIR --fig-dir DIR [--night4 ~/runs/night4]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

import analyze_val as av
from mdf import data as mdata
from mdf import explore, runinfo
from train_fit import NPIX, train_metrics, welch

CONFIGS = ("B", "MD8", "SA8")
SEEDS = (0, 1, 2)
FRACTIONS = ("f25", "f50", "full")
COMPARISONS = (("MD8", "B"), ("SA8", "B"))
METRICS = ("rel_all", "rel_p", "rel_vy")


def run_dirs(night4):
    home = Path.home() / "runs"
    full = {"B": home / "night1" / "ep133_s{s}", "MD8": home / "night2" / "MD8_s{s}",
            "SA8": home / "night2" / "SA8_s{s}"}
    out = {}
    for f in FRACTIONS:
        for c in CONFIGS:
            for s in SEEDS:
                d = Path(str(full[c]).format(s=s)) if f == "full" else night4 / f"{c}_{f}_s{s}"
                out[(f, c, s)] = d
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--night4", type=Path, default=Path.home() / "runs" / "night4")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / "runs" / "explore-cache")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=0)
    return ap.parse_args()


def msd(x, scale=1.0, nd=4):
    x = scale * np.asarray(x, dtype=float)
    return f"{x.mean():.{nd}f} ({x.std(ddof=1):.{nd}f})"


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    dirs = run_dirs(args.night4.expanduser())

    t0 = time.perf_counter()
    train_ds = explore.load_split("train", "grouped")
    val_ds = explore.load_split("val", "grouped")
    print(f"loaded train {len(train_ds)} / val {len(val_ds)} in {time.perf_counter() - t0:.1f} s")
    index = {f: i for i, f in enumerate(train_ds.files)}
    props = explore.sample_properties(val_ds)
    profiles = [p["profile"] for p in props]
    regions = explore.all_region_masks(val_ds, props)
    val_sha = mdata.file_list_sha256(val_ds.files)

    rows, per_sample, curves = [], {}, {}
    for (f, c, s), d in dirs.items():
        cfg = json.loads((d / "config.json").read_text())
        assert cfg.get("status") == "finished" and cfg["seed"] == s, d
        sp = cfg["split"]
        assert sp["split_mode"] == "grouped" and sp["val_files_sha256"] == val_sha, f"{d}: validation split differs"
        frac = sp.get("train_fraction")
        assert (frac is None) == (f == "full"), f"{d}: unexpected train_fraction {frac}"
        files = train_ds.files if frac is None else mdata.subsample_per_profile(train_ds.files, frac, sp["subset_seed"])
        assert mdata.file_list_sha256(files) == sp["train_files_sha256"], f"{d}: rebuilt training list differs"
        assert len(files) == sp["n_train"]
        idx = torch.tensor([index[x] for x in files])
        tr = train_metrics(d, train_ds.inputs[idx], train_ds.targets[idx], 10)
        r, _ = explore.cached_inference(d, val_ds, regions, [], args.cache_dir, "val")
        per_sample[(f, c, s)] = r
        with open(d / "metrics.csv") as fh:
            m = list(csv.DictReader(fh))
        val_curve = np.array([float(x["val_L1_mean_of_batches"]) for x in m])
        curves[(f, c, s)] = (val_curve, np.array([float(x["train_L1_mean_of_batches"]) for x in m]))
        val_l1 = float(np.mean(r["abs_err_all"] / NPIX))
        rows.append({"fraction": f, "n_train": len(files), "config": c, "seed": s,
                     "run_dir": str(d).replace(str(Path.home()), "~"),
                     "commit": cfg.get("git", {}).get("commit", "")[:7],
                     **{m_: float(r[m_].mean()) for m_ in METRICS},
                     "val_L1_final_metrics_csv": float(val_curve[-1]),
                     "train_L1_eval": tr["L1"], "val_L1_eval": val_l1, "gap_L1": val_l1 - tr["L1"],
                     "val_min_block": int(np.argmin(val_curve)), "n_blocks": int(val_curve.size),
                     "val_L1_min": float(val_curve.min())})
        print(f"{f} {c} s{s}: n_train {len(files)} train L1 {tr['L1']:.4e} val L1 {val_l1:.4e} "
              f"rel_all {100 * r['rel_all'].mean():.4f}", flush=True)
    av.write_csv(args.out_dir / "runs.csv", rows)

    def sel(f, c, key):
        return [r[key] for r in rows if r["fraction"] == f and r["config"] == c]

    share = {f: sel(f, "B", "n_train")[0] / len(train_ds) for f in FRACTIONS}
    summ = []
    for f in FRACTIONS:
        for c in CONFIGS:
            row = {"fraction": f"{100 * share[f]:.1f} %", "config": c, "seeds": len(sel(f, c, "seed"))}
            for m_ in METRICS:
                row[f"{m_} %"] = msd(sel(f, c, m_), 100)
            row["val_L1_final x1e-3"] = msd(sel(f, c, "val_L1_final_metrics_csv"), 1e3)
            for k in ("train_L1_eval", "val_L1_eval", "gap_L1"):
                row[f"{k} x1e-3"] = msd(sel(f, c, k), 1e3)
            row["val min block (of 133)"] = ", ".join(str(b) for b in sel(f, c, "val_min_block"))
            summ.append(row)
    av.write_csv(args.out_dir / "summary_by_group.csv", summ)

    comps = []
    for f in FRACTIONS:
        for v, ref in COMPARISONS:
            for m_ in METRICS:
                mv = np.stack([per_sample[(f, v, s)][m_] for s in SEEDS])
                mr = np.stack([per_sample[(f, ref, s)][m_] for s in SEEDS])
                b = explore.paired_bootstrap(mv, mr, profiles, args.n_boot, args.boot_seed)
                comps.append({"fraction": f"{100 * share[f]:.1f} %", "comparison": f"{v} - {ref}", "metric": m_,
                              "diff (points)": f"{100 * b['mean_diff']:+.4f}",
                              "95% CI seeds+profiles": f"[{100 * b['ci2_lo']:+.4f}, {100 * b['ci2_hi']:+.4f}]",
                              "95% CI profiles": f"[{100 * b['ci1_lo']:+.4f}, {100 * b['ci1_hi']:+.4f}]"})
            for k in ("train_L1_eval", "val_L1_eval", "gap_L1"):
                dd, se, z = welch(sel(f, v, k), sel(f, ref, k))
                comps.append({"fraction": f"{100 * share[f]:.1f} %", "comparison": f"{v} - {ref}",
                              "metric": f"{k} x1e-3", "diff (points)": f"{1e3 * dd:+.4f}",
                              "95% CI seeds+profiles": f"diff/SE {z:+.2f}", "95% CI profiles": ""})
    av.write_csv(args.out_dir / "comparisons.csv", comps)

    # --- figures ---
    x = np.array([100 * share[f] for f in FRACTIONS])
    panels = [("rel_all", 100, "rel_all (%)"), ("rel_p", 100, "rel_p (%)"), ("rel_vy", 100, "rel_vy (%)"),
              ("train_L1_eval", 1e3, "train L1, eval mode (x1e-3)"), ("val_L1_eval", 1e3, "val L1 (x1e-3)"),
              ("gap_L1", 1e3, "gap val - train L1 (x1e-3)")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    offs = {"B": -1.2, "MD8": 0.0, "SA8": 1.2}
    for ax, (k, sc, lab) in zip(axes.flat, panels):
        for c in CONFIGS:
            vals = np.array([sc * np.array(sel(f, c, k)) for f in FRACTIONS])  # (F, 3)
            ax.errorbar(x + offs[c], vals.mean(1), yerr=vals.std(1, ddof=1), marker="o", capsize=3, label=c)
            ax.scatter(np.repeat(x + offs[c], vals.shape[1]), vals.ravel(), s=8, alpha=0.4)
        ax.set_xlabel("training samples (% of 5998)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("Data-quantity experiment (validation, EXPLORATORY): mean +- sd over seeds 0-2, dots = runs")
    fig.tight_layout()
    fig.savefig(args.fig_dir / "metric_vs_fraction.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, c in zip(axes, CONFIGS):
        for f, col in zip(FRACTIONS, ("C0", "C1", "C2")):
            for s in SEEDS:
                vc, _ = curves[(f, c, s)]
                ax.plot(np.arange(vc.size), 1e3 * vc, color=col, lw=0.8, alpha=0.8,
                        label=f"{100 * share[f]:.0f} %" if s == 0 else None)
                ax.plot(np.argmin(vc), 1e3 * vc.min(), "v", color=col, ms=4)
        ax.set_title(c)
        ax.set_xlabel("validation block (599 steps; = epoch for 100 %)")
        ax.set_ylim(4, 8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("val L1 (metrics.csv, x1e-3)")
    axes[0].legend(title="training data")
    fig.suptitle("Validation curves (diagnostic only; triangles = minimum, NOT used as a result)")
    fig.tight_layout()
    fig.savefig(args.fig_dir / "val_curves.png", dpi=120)
    plt.close(fig)

    md = ["# Data-quantity experiment (validation only, EXPLORATORY)", "",
          f"Generated by `scripts/data_quantity.py`, git {runinfo.git_info()}, {runinfo.utc_now()}. Pre-registered in "
          "commit 26b6e7f (section 1). Validation: "
          f"{len(val_ds)} samples, {len(set(profiles))} profiles. Bootstrap {args.n_boot}, seed {args.boot_seed}. "
          "**3 seeds per group: intervals and SEs are rough.** Main number = final checkpoint.", "",
          "Training lists rebuilt from the subset rule; sha256 equal to every run's config.json (asserted).", "",
          "## Per group: mean (sd over seeds 0-2)", "", av.md_table(summ, list(summ[0].keys())), "",
          "## Differences per fraction", "",
          "rel_*: per-sample paired difference, two-level bootstrap (seeds + profiles) and one-level (profiles); "
          "L1 rows: Welch diff / SE over seeds.", "", av.md_table(comps, list(comps[0].keys())), "",
          "Figures: `metric_vs_fraction.png`, `val_curves.png`.", ""]
    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
