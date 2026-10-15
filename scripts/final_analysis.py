"""Pre-registered final analysis (docs/lab-notebook.md, 2026-10-14, "Decisions ... before looking at any metric of
seeds 3-7", section 2), runnable on validation (exploratory) and, after the explicit go-ahead, on the test set.

The test split is refused unless --split test --allow-test is given (mdf/explore.py guard). No strata here.

Contents
  T1  per configuration, mean +- std (ddof=1) over seeds of the per-run values: rel_all, rel_p, rel_vx, rel_vy,
      aggregate rel_p (sum|e| / sum|t| over all samples), median rel_p, MAE in the fluid per channel
  T2  paired comparisons var - ref: per-sample metric, two-level bootstrap (seeds + profiles, --n-boot repetitions),
      also the one-level CI (profiles); metrics rel_all, rel_p, rel_vx, rel_vy and fluid MAE per channel
      main: MD8 - B, MD8 - SA8, MD8 - B-large; exploratory: MD32 - MD8, SA32 - SA8, MD32 - SA32, CBAM8 - B;
      second pass (if the label is loaded): MDv8 - MD8, MDv8 - B
      position (night 5, exploratory, validation only; if the labels are loaded): MDv32 - MD32, MDv32 - B,
      MDv64 - B, MDv8-16-32 - MDv8, MDv8-16-32 - B
      confirm-mdv64 (only with --confirm-mdv64; pre-registration commit d46b279): MDv64 - B. Decision on rel_vy only:
      "CONFIRMED" if the upper bound of the two-level 95 % interval (10 000 repetitions) is < 0, else "NOT CONFIRMED";
      printed first in summary.md with the descriptive MDv64 - B rows of T2 / T2b / T3 (wall vy) / T4
  T3  regions (descriptive): MAE per region and channel, differences with the two-level CI
  T4  divergence (descriptive; mdf/explore.py DIV_MAPPING): mean |div pred|, mean |div target|,
      mean |div pred - div target| in the wall band and in the other fluid cells; differences of the last one

Per-sample values come from explore.cached_inference (batch 1, as evaluate.py); a consistency check against each
run's eval_<split>_samples.csv is printed when that file exists.

Usage:
    python scripts/final_analysis.py --config B=GLOB [--config ...] [--seeds 0-7] [--split val]
                                     [--out-dir DIR] [--fig-dir DIR] [--confirm-mdv64]
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `mdf`
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/, for analyze_val helpers

import matplotlib.pyplot as plt
import numpy as np
import torch

import analyze_val as av
from mdf import data as mdata
from mdf import explore, runinfo

REPO = Path(__file__).resolve().parent.parent
MAIN = [("MD8", "B"), ("MD8", "SA8"), ("MD8", "B-large")]
EXPLORATORY = [("MD32", "MD8"), ("SA32", "SA8"), ("MD32", "SA32"), ("CBAM8", "B")]
SECOND_PASS = [("MDv8", "MD8"), ("MDv8", "B")]
# position experiment (notebook 2026-10-15, pre-registration commit 72e5b1b; validation only)
POSITION = [("MDv32", "MD32"), ("MDv32", "B"), ("MDv64", "B"), ("MDv8-16-32", "MDv8"), ("MDv8-16-32", "B")]
# confirmatory test of MDv64 (notebook 2026-10-15, pre-registration commit d46b279; only with --confirm-mdv64)
CONFIRM_MDV64 = [("MDv64", "B")]
CONFIRM_METRIC, CONFIRM_N_BOOT, CONFIRM_BOOT_SEED = "rel_vy", 10000, 0
ORDER = ["B", "MD8", "SA8", "B-large", "CBAM8", "MD32", "SA32", "MDv8", "MDv32", "MDv64", "MDv8-16-32"]
CH = explore.CHANNEL_NAMES
FLUID = ("wall", "wake", "rest")
# aggregate and median ratios: p (pre-registered, commit 26b6e7f) and vy (addendum of 2026-10-14, decided after the
# 8-seed validation analysis and before the test set was opened)
AGG_MEDIAN_CHANNELS = ("p", "vy")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", action="append", required=True, help="LABEL=GLOB of run directories (repeatable)")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true", help="required with --split test (final results only)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / "runs" / "explore-cache")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--confirm-mdv64", action="store_true",
                    help="add the pre-registered confirmatory comparison MDv64 - B (decision on rel_vy)")
    args = ap.parse_args()
    if args.confirm_mdv64 and (args.n_boot, args.boot_seed) != (CONFIRM_N_BOOT, CONFIRM_BOOT_SEED):
        ap.error(f"--confirm-mdv64 is pre-registered with --n-boot {CONFIRM_N_BOOT} --boot-seed {CONFIRM_BOOT_SEED}")
    return args


def msd(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), (x.std(ddof=1) if x.size > 1 else float("nan"))


def fmt_msd(x, scale=100.0, nd=4):
    m, s = msd(np.asarray(x) * scale)
    return f"{m:.{nd}f} ({s:.{nd}f})"


def per_sample_fluid_mae(r):
    """(N, 3) per-sample mean |o - t| over the fluid cells."""
    err = sum(r[f"reg_err_{g}"] for g in FLUID)
    cnt = sum(r[f"reg_n_{g}"] for g in FLUID)
    return err / cnt[:, None]


def main():
    args = parse_args()
    runs = av.collect_runs(args.config, av.parse_seeds(args.seeds))
    args._seeds = {l: [s for s, _, _ in lst] for l, lst in runs.items()}
    labels = [l for l in ORDER if l in runs] + [l for l in runs if l not in ORDER]
    if args.confirm_mdv64 and not {"MDv64", "B"} <= set(runs):
        sys.exit("--confirm-mdv64 needs the labels MDv64 and B")
    groups = [("main", MAIN), ("exploratory", EXPLORATORY), ("second pass", SECOND_PASS), ("position", POSITION)]
    if args.confirm_mdv64:
        groups.append(("confirm-mdv64", CONFIRM_MDV64))
    all_pairs = list(dict.fromkeys(pair for _, pairs in groups for pair in pairs))
    split_modes = {cfg.get("split", {}).get("split_mode", "random") for lst in runs.values() for _, _, cfg in lst}
    if split_modes != {"grouped"}:
        sys.exit(f"runs use different / non-grouped splits: {split_modes}")
    ds = explore.load_split(args.split, "grouped", allow_test=args.allow_test)
    if args.split == "val":
        shas = {cfg["split"]["val_files_sha256"] for lst in runs.values() for _, _, cfg in lst}
        if shas != {mdata.file_list_sha256(ds.files)}:
            sys.exit("validation file list differs from the one recorded in the runs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    props = explore.sample_properties(ds)
    profiles = [p["profile"] for p in props]
    regions = explore.all_region_masks(ds, props)

    res, run_rows, max_diff = {}, [], 0.0
    for label in labels:
        res[label] = []
        for seed, d, cfg in runs[label]:
            r, cached = explore.cached_inference(d, ds, regions, [], args.cache_dir, args.split, args.device)
            res[label].append(r)
            ev = d / f"eval_{args.split}_samples.csv"
            diff = float("nan")
            if ev.exists():
                with open(ev) as fh:
                    rows = list(csv.DictReader(fh))
                assert [row["file"] for row in rows] == ds.files, f"{ev}: file order differs"
                diff = max(float(np.max(np.abs(np.array([float(row[m]) for row in rows]) - r[m])))
                           for m in av.METRICS)
                max_diff = max(max_diff, diff)
            run_rows.append({"label": label, "seed": seed, "run_dir": str(d).replace(str(Path.home()), "~"),
                             "commit": cfg.get("git", {}).get("commit", "")[:7],
                             "max_abs_diff_vs_eval_csv": diff})
            print(f"{label} s{seed}: {'cache' if cached else 'inference'}; max |diff| vs eval CSV {diff:.3g}")
    av.write_csv(args.out_dir / "runs.csv", run_rows)

    def stack(label, key):
        return np.stack([r[key] for r in res[label]])

    md = [f"# Final analysis on split `{args.split}` (pre-registered plan, commit 26b6e7f)", "",
          f"Generated by `scripts/final_analysis.py`, git {runinfo.git_info()}, {runinfo.utc_now()}. "
          f"{len(ds)} samples, {len(set(profiles))} profiles. Bootstrap {args.n_boot} repetitions, seed "
          f"{args.boot_seed}. Values in % unless stated.", "",
          f"Consistency: max |recomputed - eval CSV| over all runs and {av.METRICS}: **{max_diff:.3g}** (ratios).", ""]

    # --- T1 ---
    t1 = []
    for label in labels:
        row = {"config": label, "seeds": len(res[label])}
        for m in av.METRICS:
            row[m] = fmt_msd(stack(label, m).mean(axis=1))
        for c in AGG_MEDIAN_CHANNELS:
            row[f"agg_rel_{c}"] = fmt_msd(stack(label, f"abs_err_{c}").sum(1) / stack(label, f"abs_tgt_{c}").sum(1))
            row[f"median_rel_{c}"] = fmt_msd(np.median(stack(label, f"rel_{c}"), axis=1))
        mae = np.stack([per_sample_fluid_mae(r).mean(0) for r in res[label]])  # (S, 3)
        for i, c in enumerate(CH):
            row[f"mae_fluid_{c} x1e-3"] = fmt_msd(mae[:, i], scale=1e3)
        t1.append(row)
    av.write_csv(args.out_dir / "t1_summary.csv", t1)
    md += ["## T1 Mean (std over seeds) per configuration", "",
           "rel_* = mean over samples of the per-sample ratio, then mean (std, ddof=1) over seeds; agg_rel_c = "
           "sum|e_c| / sum|t_c| over all samples; median_rel_c = median over samples (c = p, vy); MAE in the fluid (normalised "
           "fields, x1e-3) = per-sample mean |o - t| over fluid cells, mean over samples.", "",
           av.md_table(t1, list(t1[0].keys())), ""]

    # --- T2 ---
    t2 = []
    fluid_mae = {l: np.stack([per_sample_fluid_mae(r) for r in res[l]]) for l in labels}  # (S, N, 3)
    for group, pairs in groups:
        for v, ref in pairs:
            if v not in res or ref not in res:
                continue
            metrics = [(m, stack(v, m), stack(ref, m), 100.0) for m in av.METRICS]
            metrics += [(f"mae_fluid_{c} x1e-3", fluid_mae[v][..., i], fluid_mae[ref][..., i], 1e3)
                        for i, c in enumerate(CH)]
            for name, a, b, sc in metrics:
                bs = explore.paired_bootstrap(a, b, profiles, args.n_boot, args.boot_seed)
                t2.append({"group": group, "comparison": f"{v} - {ref}", "metric": name, "scale": sc, **bs})
    av.write_csv(args.out_dir / "t2_paired.csv", t2)
    t2b = []
    for group, pairs in groups:
        for v, ref in pairs:
            if v not in res or ref not in res:
                continue
            for c in AGG_MEDIAN_CHANNELS:
                tgt = res[ref][0][f"abs_tgt_{c}"]
                assert np.allclose(tgt, res[v][0][f"abs_tgt_{c}"])
                bs = explore.paired_bootstrap_agg_median(stack(v, f"abs_err_{c}"), stack(ref, f"abs_err_{c}"), tgt,
                                                         stack(v, f"rel_{c}"), stack(ref, f"rel_{c}"), profiles,
                                                         args.n_boot, args.boot_seed)
                t2b.append({"group": group, "comparison": f"{v} - {ref}", "channel": c, **bs})
    av.write_csv(args.out_dir / "t2_agg_median.csv", t2b)
    rows_b = []
    for r in t2b:
        for k in ("agg", "median"):
            rows_b.append({"group": r["group"], "comparison": r["comparison"], "metric": f"{k}_rel_{r['channel']}",
                           "seeds": f"{r['seeds_var']}/{r['seeds_ref']}",
                           "diff": f"{100 * r[k + '_diff']:+.4f}",
                           "95% CI seeds+profiles": f"[{100 * r[k + '_ci2_lo']:+.4f}, {100 * r[k + '_ci2_hi']:+.4f}]",
                           "95% CI profiles": f"[{100 * r[k + '_ci1_lo']:+.4f}, {100 * r[k + '_ci1_hi']:+.4f}]"})
    rows_md = [{"group": r["group"], "comparison": r["comparison"], "metric": r["metric"],
                "seeds": f"{r['seeds_var']}/{r['seeds_ref']}",
                "mean diff": f"{r['scale'] * r['mean_diff']:+.4f}",
                "95% CI seeds+profiles": f"[{r['scale'] * r['ci2_lo']:+.4f}, {r['scale'] * r['ci2_hi']:+.4f}]",
                "95% CI profiles": f"[{r['scale'] * r['ci1_lo']:+.4f}, {r['scale'] * r['ci1_hi']:+.4f}]",
                "median diff": f"{r['scale'] * r['median_diff']:+.4f}",
                "profiles better": f"{r['frac_profiles_var_better']:.2f}"} for r in t2]
    md += ["## T2 Paired comparisons (negative = variant has lower error)", "",
           "Per-sample seed means, difference per sample; main interval: two-level bootstrap (seeds of each "
           "configuration + profiles). rel_* in percentage points, MAE x1e-3. No correction for multiple "
           "comparisons; descriptive.", "", av.md_table(rows_md, list(rows_md[0].keys())) if rows_md else "", "",
           "### T2b Aggregate and median ratios (p pre-registered; vy added by the 2026-10-14 addendum)", "",
           "agg diff = difference of sum|e| / sum|t| over all samples (seed-mean errors); median diff = difference of "
           "the medians over samples of the seed-mean per-sample ratio; percentage points; bootstrap as T2 (a sample "
           "counts as many times as its profile is drawn).", "",
           av.md_table(rows_b, list(rows_b[0].keys())) if rows_b else "", ""]

    # --- T3 regions ---
    t3, t3d = [], []
    reg_mae = {}
    for label in labels:
        for g in explore.REGIONS:
            e = stack(label, f"reg_err_{g}") / np.maximum(stack(label, f"reg_n_{g}"), 1)[..., None]  # (S, N, 3)
            reg_mae[(label, g)] = e
            m = e.mean(axis=(0, 1))
            t3.append({"config": label, "region": g, **{f"mae_{c} x1e-3": f"{1e3 * m[i]:.4f}" for i, c in enumerate(CH)}})
    for v, ref in all_pairs:
        if v not in res or ref not in res:
            continue
        for g in ("wall", "wake", "rest"):
            for i, c in enumerate(CH):
                bs = explore.paired_bootstrap(reg_mae[(v, g)][..., i], reg_mae[(ref, g)][..., i], profiles,
                                              args.n_boot, args.boot_seed)
                t3d.append({"comparison": f"{v} - {ref}", "region": g, "channel": c,
                            "diff x1e-3": f"{1e3 * bs['mean_diff']:+.4f} [{1e3 * bs['ci2_lo']:+.4f}, "
                                          f"{1e3 * bs['ci2_hi']:+.4f}]",
                            "rel_change": f"{100 * bs['mean_diff'] / reg_mae[(ref, g)][..., i].mean():+.1f} %"})
    av.write_csv(args.out_dir / "t3_regions.csv", t3)
    av.write_csv(args.out_dir / "t3_region_diffs.csv", t3d)
    cells = {g: np.mean([reg[g].sum() for reg in regions]) for g in explore.REGIONS}
    md += ["## T3 Regions (descriptive; MAE x1e-3, normalised fields)", "",
           f"wall = fluid cells 1-{explore.WALL_PX} px from the airfoil; wake = downstream of the trailing edge within "
           f"{explore.WAKE_HALF_WIDTH} px of the free-stream line; rest = other fluid; interior = airfoil. Mean cells "
           "per sample: " + ", ".join(f"{g} {cells[g]:.0f}" for g in explore.REGIONS) + ".", "",
           av.md_table(t3, list(t3[0].keys())), "", "Differences, two-level 95% CI:", "",
           av.md_table(t3d, list(t3d[0].keys())) if t3d else "", ""]

    # --- T4 divergence ---
    t4, t4d = [], []
    for label in labels:
        row = {"config": label, "seeds": len(res[label])}
        for g in explore.DIV_REGIONS:
            for k in ("pred", "tgt", "err"):
                row[f"{g} |div {k}| x1e-3"] = fmt_msd(stack(label, f"div_{k}_{g}").mean(1), scale=1e3)
        t4.append(row)
    for v, ref in all_pairs:
        if v not in res or ref not in res:
            continue
        for g in explore.DIV_REGIONS:
            bs = explore.paired_bootstrap(stack(v, f"div_err_{g}"), stack(ref, f"div_err_{g}"), profiles,
                                          args.n_boot, args.boot_seed)
            t4d.append({"comparison": f"{v} - {ref}", "region": g,
                        "diff |div pred - div tgt| x1e-3": f"{1e3 * bs['mean_diff']:+.4f} [{1e3 * bs['ci2_lo']:+.4f}, "
                                                          f"{1e3 * bs['ci2_hi']:+.4f}]"})
    av.write_csv(args.out_dir / "t4_divergence.csv", t4)
    av.write_csv(args.out_dir / "t4_divergence_diffs.csv", t4d)
    n_div = {g: float(np.mean(res[labels[0]][0][f"div_n_{g}"])) for g in explore.DIV_REGIONS}
    md += ["## T4 Divergence (descriptive)", "",
           f"Mapping {explore.DIV_MAPPING} (vx along array axis 0, same sign); v / |v_inf| units, per pixel, central "
           "differences, fluid cells with 4 fluid neighbours. Per sample: mean over cells, then mean over samples, "
           "mean (std) over seeds. Regions: wall = wall band, other = wake + rest. Mean valid cells per sample: "
           + ", ".join(f"{g} {n_div[g]:.0f}" for g in explore.DIV_REGIONS)
           + ". The target is not exactly divergence-free: compare with the target, not with 0.", "",
           av.md_table(t4, list(t4[0].keys())), "", "Differences of mean |div pred - div target| (two-level CI):", "",
           av.md_table(t4d, list(t4d[0].keys())) if t4d else "", ""]

    # --- figure: per-seed values ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, m in zip(axes, ["rel_all", "rel_p"]):
        for j, label in enumerate(labels):
            vals = 100 * stack(label, m).mean(axis=1)
            ax.scatter(np.full(vals.size, j), vals, s=18, color="0.6", zorder=2)
            ax.scatter([j], [vals.mean()], s=60, marker="_", color="k", linewidths=2, zorder=3)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_ylabel(f"{m} (%)")
        ax.set_title(f"{m}: seeds (grey), mean (black)", fontsize=9)
        ax.grid(axis="y", color="0.9")
    fig.suptitle(f"Per-run {args.split} error per configuration", fontsize=9)
    fig.savefig(args.fig_dir / "per_seed.png", dpi=120)
    plt.close(fig)

    if args.confirm_mdv64:
        md[md.index("## T1 Mean (std over seeds) per configuration"):md.index("## T1 Mean (std over seeds) per configuration")] \
            = confirm_section(args, res, t2, rows_md, rows_b, t3d, t4d)
    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {args.out_dir}/summary.md")


def confirm_section(args, res, t2, rows_md, rows_b, t3d, t4d):
    """Pre-registered decision (commit d46b279) and the descriptive MDv64 - B rows; also printed to stdout."""
    comp = "MDv64 - B"
    (r,) = [x for x in t2 if x["group"] == "confirm-mdv64" and x["metric"] == CONFIRM_METRIC]
    confirmed = r["ci2_hi"] < 0
    verdict = "CONFIRMED" if confirmed else "NOT CONFIRMED"
    seeds = args._seeds
    line = (f"{comp} {CONFIRM_METRIC}: {100 * r['mean_diff']:+.4f} points, two-level 95 % CI "
            f"[{100 * r['ci2_lo']:+.4f}, {100 * r['ci2_hi']:+.4f}] ({CONFIRM_N_BOOT} repetitions, seed "
            f"{CONFIRM_BOOT_SEED}); seeds MDv64 {seeds['MDv64']}, B {seeds['B']}; split {args.split}")
    banner = "=" * 100
    print(f"\n{banner}\nCONFIRMATORY TEST MDv64 vs B (pre-registered, commit d46b279): **{verdict}**\n{line}\n"
          f"Rule: CONFIRMED only if the upper bound of the interval is < 0."
          + ("" if args.split == "test" else "\nNOT THE TEST SET: functional check only, no decision.") + f"\n{banner}\n")
    desc_t2 = [x for x in rows_md if x["group"] == "confirm-mdv64"]
    desc_t2b = [x for x in rows_b if x["group"] == "confirm-mdv64"]
    desc_t3 = [x for x in t3d if x["comparison"] == comp and x["region"] == "wall" and x["channel"] == "vy"]
    desc_t4 = [x for x in t4d if x["comparison"] == comp]
    out = [f"## CONFIRMATORY TEST: MDv64 vs B on rel_vy (pre-registered, commit d46b279) — **{verdict}**", "",
           f"**{line}.** Rule: CONFIRMED only if the upper bound of the two-level 95 % interval is < 0; no other metric "
           "decides." + ("" if args.split == "test" else " **Not the test set: functional check only, no decision.**"),
           "", "Descriptive only (no decision), MDv64 - B:", "",
           av.md_table(desc_t2, list(desc_t2[0].keys())), "", av.md_table(desc_t2b, list(desc_t2b[0].keys())), "",
           "Wall band, vy MAE x1e-3:", "", av.md_table(desc_t3, list(desc_t3[0].keys())), "",
           "Divergence error:", "", av.md_table(desc_t4, list(desc_t4[0].keys())), ""]
    return out


if __name__ == "__main__":
    main()
