"""Exploratory per-sample analysis of trained runs on one split (A1-A7, docs/lab-notebook.md 2026-10-13).

EXPLORATORY: all outputs are descriptive; the test split is for final results only and is refused unless
--split test --allow-test is given explicitly (and every loaded path is asserted to be outside data/test for
split val, mdf/explore.py).

Runs are chosen on the command line, so more seeds or other run roots need no code change:
    --config LABEL=GLOB   (repeatable; the same LABEL may be given several times, e.g. B from two nights)
    --seeds 0-7           (seed read from each run's config.json; default: all found)
Without --config the night-1 / night-2 runs of 2026-10-11..13 are used (labels B, MD8, SA8, MD32, SA32,
B-large, CBAM8).

Steps
  0  inference of every run (GPU by default, a few seconds per run; --device cpu uses the Mamba reference
     path), cached in --cache-dir; per-sample sums, per-region sums, mean |error| map and the predictions of
     the A7 samples only. Consistency check against the run's eval_<split>_samples.csv.
  A1 channel weights of rel_all and decomposition of rel_all differences into channel contributions
  A2 mean absolute error per channel (normalised fields), fluid cells and airfoil interior
  A3 paired per-sample comparisons, bootstrap over profiles (one level) and seeds + profiles (two levels)
  A4 differences by quintile of difficulty criteria that do not use the errors of the compared models
  A5 mean |error| maps and errors by region (wall band, wake, rest, interior)
  A6 train / validation L1 at the last epoch and mean validation curves (metrics.csv)
  A7 qualitative figures (high |alpha| and small pressure signal) and a flow-separation view

Outputs: --out-dir (CSV tables + summary.md), --fig-dir (PNG).

Usage:
    python scripts/analyze_val.py [--config B=~/runs/night1/ep133_s* ...] [--seeds 0-2]
                                  [--split val] [--out-dir docs/results/val-explore]
                                  [--fig-dir docs/figures/val-explore] [--device cuda]
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `mdf`

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mdf import explore, runinfo

REPO = Path(__file__).resolve().parent.parent
HOME_RUNS = Path.home() / "runs"
DEFAULT_CONFIGS = [
    f"B={HOME_RUNS}/night1/ep133_s*", f"MD8={HOME_RUNS}/night2/MD8_s*", f"SA8={HOME_RUNS}/night2/SA8_s*",
    f"MD32={HOME_RUNS}/night2/MD32_s*", f"SA32={HOME_RUNS}/night2/SA32_s*",
    f"B-large={HOME_RUNS}/night2/B-large_s*", f"CBAM8={HOME_RUNS}/night2/CBAM8_s*",
]
COMPARISONS = [("MD8", "B"), ("MD32", "B"), ("SA8", "B"), ("SA32", "B"), ("B-large", "B"), ("CBAM8", "B"),
               ("MD8", "SA8"), ("MD32", "SA32"), ("MD32", "MD8"), ("MD8", "B-large"), ("SA32", "SA8"),
               ("MDv8", "MD8"), ("MDv8", "B")]  # pairs with a missing label are skipped
STRATA_COMPARISONS = [("MD8", "B"), ("MD32", "B"), ("SA8", "B"), ("SA32", "B"), ("MD8", "SA8")]
METRICS = ["rel_all", "rel_p", "rel_vx", "rel_vy"]
CH = explore.CHANNEL_NAMES


def parse_seeds(text):
    if text is None:
        return None
    out = set()
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", action="append", help="LABEL=GLOB of run directories (repeatable)")
    ap.add_argument("--seeds", default=None, help="e.g. 0-2 or 0,1,2,5 (default: all runs found)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true", help="required with --split test (final results only)")
    ap.add_argument("--out-dir", type=Path, default=REPO / "docs/results/val-explore")
    ap.add_argument("--fig-dir", type=Path, default=REPO / "docs/figures/val-explore")
    ap.add_argument("--cache-dir", type=Path, default=HOME_RUNS / "explore-cache")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--qual-seed", type=int, default=0, help="seed of the runs shown in the A7 figures")
    ap.add_argument("--qual-files", default="pfcm_6703_-2724.npz,rae5214_8305_-3198.npz",
                    help="known hard high-angle cases shown in A7 if present in the split; filled up to 3 by |alpha|")
    ap.add_argument("--external", default="B-large,CBAM8",
                    help="labels whose seed-mean rel_p defines the external difficulty (A4 d)")
    ap.add_argument("--external-seeds", default=None,
                    help="seeds of the --external labels used for the external difficulty (default: all loaded)")
    ap.add_argument("--previous", type=Path, default=None,
                    help="out-dir of an earlier run of this script: adds an 'earlier vs now' section (A3, A4 |alpha|)")
    return ap.parse_args()


# --- runs ---

def collect_runs(config_specs, seeds):
    runs = {}  # label -> list of (seed, run_dir, cfg)
    for spec in config_specs:
        label, _, pattern = spec.partition("=")
        if not pattern:
            sys.exit(f"bad --config {spec!r}, expected LABEL=GLOB")
        for d in sorted(glob.glob(os.path.expanduser(pattern))):
            d = Path(d)
            if not (d / "config.json").exists():
                continue
            cfg = json.loads((d / "config.json").read_text())
            if cfg.get("status") != "finished" or (seeds is not None and cfg["seed"] not in seeds):
                continue
            runs.setdefault(label, []).append((cfg["seed"], d, cfg))
    for label, lst in runs.items():
        lst.sort(key=lambda r: r[0])
        s = [r[0] for r in lst]
        if len(s) != len(set(s)):
            sys.exit(f"{label}: duplicate seeds {s}")
    return runs


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.8g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


def md_table(rows, fields, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for r in rows:
        cells = []
        for f in fields:
            v = r[f]
            cells.append(fmt[f](v) if f in fmt else (f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pct(v):
    return f"{100 * v:+.3f}" if v < 0 or v > 0 else "0"


def _status(prev_excl, now_excl):
    return {(True, True): "holds", (True, False): "disappeared", (False, True): "appeared",
            (False, False): "none"}[(prev_excl, now_excl)]


def earlier_vs_now(prev_dir, a3, a4, out_dir):
    """Earlier run (e.g. seeds 0-2) vs this run, per comparison: estimate and two-level CI. Rule: an indication
    'holds' if the two-level CI excludes 0 in both, 'disappeared' / 'appeared' if only in the earlier / current one."""
    with open(prev_dir / "a3_paired.csv") as fh:
        p3 = {(r["comparison"], r["metric"]): r for r in csv.DictReader(fh)}
    with open(prev_dir / "a4_strata.csv") as fh:
        p4 = {(r["comparison"], r["metric"], r["quintile"]): r for r in csv.DictReader(fh)
              if r["criterion"] == "abs_alpha"}
    rows = []

    def add(kind, key, prev, now_d, lo, hi, seeds):
        pl, ph = float(prev["ci2_lo"]), float(prev["ci2_hi"])
        rows.append({"what": kind, "comparison": key[0], "metric": key[1], "seeds now": seeds,
                     "earlier diff [CI2]": f"{100 * float(prev['mean_diff']):+.3f} [{100 * pl:+.3f}, {100 * ph:+.3f}]",
                     "now diff [CI2]": f"{100 * now_d:+.3f} [{100 * lo:+.3f}, {100 * hi:+.3f}]",
                     "status": _status(pl > 0 or ph < 0, lo > 0 or hi < 0)})
    for r in a3:
        k = (r["comparison"], r["metric"])
        if k in p3:
            add("A3", k, p3[k], r["mean_diff"], r["ci2_lo"], r["ci2_hi"], f"{r['seeds_var']}/{r['seeds_ref']}")
    for r in a4:
        k = (r["comparison"], r["metric"], str(r["quintile"]))
        if r["criterion"] == "abs_alpha" and k in p4:
            add(f"A4 |alpha| Q{r['quintile']}", k, p4[k], r["mean_diff"], r["ci2_lo"], r["ci2_hi"], "")
    write_csv(out_dir / "earlier_vs_now.csv", rows)
    return ["## Earlier run vs this run (exploratory)", "",
            f"Earlier: `{prev_dir}`. CI2 = two-level bootstrap (seeds + profiles), percentage points. Status: holds = "
            "CI2 excludes 0 in both; disappeared / appeared = only earlier / only now; none = neither.", "",
            md_table(rows, list(rows[0].keys())) if rows else "(no common rows)", ""]


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    runs = collect_runs(args.config or DEFAULT_CONFIGS, seeds)
    if "B" not in runs:
        sys.exit("label B (reference) not found")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    md = [f"# Exploratory analysis on split `{args.split}` (EXPLORATORY, descriptive)", "",
          f"Generated by `scripts/analyze_val.py`, git {runinfo.git_info()}, "
          f"{runinfo.utc_now()}. Bootstrap: {args.n_boot} repetitions, seed {args.boot_seed}.", ""]

    # every run must have been trained / evaluated on the same split
    split_modes = {cfg.get("split", {}).get("split_mode", "random") for lst in runs.values() for _, _, cfg in lst}
    if args.split == "val" and split_modes != {"grouped"}:
        sys.exit(f"runs use different / non-grouped validation splits: {split_modes}")
    ds = explore.load_split(args.split, "grouped", allow_test=args.allow_test)
    if args.split == "val":
        shas = {cfg["split"]["val_files_sha256"] for lst in runs.values() for _, _, cfg in lst}
        from mdf import data as mdata
        if shas != {mdata.file_list_sha256(ds.files)}:
            sys.exit("validation file list differs from the one recorded in the runs")
    n = len(ds)
    props = explore.sample_properties(ds)
    profiles = [p["profile"] for p in props]
    regions = explore.all_region_masks(ds, props)
    alpha = np.array([p["abs_alpha_deg"] for p in props])
    psum = np.array([p["sum_abs_p_tgt"] for p in props])
    thick = np.array([p["thickness"] for p in props])

    # A7 samples (data-only selection): the named hard cases if present, filled up to 3 by largest |alpha|;
    # then the 3 smallest sum|p target|
    named = [ds.files.index(f) for f in args.qual_files.split(",") if f in ds.files]
    hi_alpha = (named + [int(i) for i in np.argsort(-alpha) if i not in named])[:3]
    lo_p = [i for i in np.argsort(psum) if i not in hi_alpha][:3]
    keep = [int(i) for i in hi_alpha + lo_p]

    # --- step 0: inference + consistency check ---
    res = {}  # label -> list of result dicts (same order as runs[label])
    run_rows, max_diff = [], 0.0
    for label, lst in runs.items():
        res[label] = []
        for seed, d, cfg in lst:
            r, cached = explore.cached_inference(d, ds, regions, keep, args.cache_dir, args.split, args.device)
            res[label].append(r)
            ev = d / f"eval_{args.split}_samples.csv"
            diff = float("nan")
            if ev.exists():
                with open(ev) as fh:
                    rows = list(csv.DictReader(fh))
                assert [row["file"] for row in rows] == ds.files, f"{ev}: file order differs"
                diff = max(float(np.max(np.abs(np.array([float(row[m]) for row in rows]) - r[m])))
                           for m in METRICS)
                max_diff = max(max_diff, diff)
            run_rows.append({"label": label, "seed": seed, "run_dir": str(d).replace(str(Path.home()), "~"),
                             "commit": cfg.get("git", {}).get("commit", "")[:7], "cached": cached,
                             "max_abs_diff_vs_eval_csv": diff})
            print(f"{label} s{seed}: {'cache' if cached else 'inference'}; max |diff| vs eval CSV {diff:.3g}")
    write_csv(args.out_dir / "runs.csv", run_rows)
    write_csv(args.out_dir / "sample_properties.csv", props)
    md += ["## Runs and consistency check", "",
           md_table(run_rows, ["label", "seed", "run_dir", "commit", "max_abs_diff_vs_eval_csv"]), "",
           f"Max |recomputed - eval CSV| over all runs, samples and {METRICS}: **{max_diff:.3g}** (ratios, not %).",
           ""]
    labels = list(runs)

    def stack(label, key):
        return np.stack([r[key] for r in res[label]])  # (seeds, N, ...)

    # --- A1: channel weights and decomposition ---
    r0 = res["B"][0]
    w = np.stack([r0[f"abs_tgt_{c}"] for c in CH]) / r0["abs_tgt_all"]  # (3, N), target only
    ident = max(float(np.max(np.abs((w * np.stack([rr[f"rel_{c}"] for c in CH])).sum(0) - rr["rel_all"])))
                for lst in res.values() for rr in lst)
    a1 = [{"channel": c, "w_mean": w[i].mean(), "w_min": w[i].min(), "w_max": w[i].max()} for i, c in enumerate(CH)]
    write_csv(args.out_dir / "a1_channel_weights.csv", a1)
    dec = []
    for v in labels:
        if v == "B":
            continue
        row = {"comparison": f"{v} - B", "d_rel_all": float(stack(v, "rel_all").mean() - stack("B", "rel_all").mean())}
        for i, c in enumerate(CH):
            row[f"contrib_{c}"] = float(np.mean(w[i] * (stack(v, f"rel_{c}").mean(0) - stack("B", f"rel_{c}").mean(0))))
            row[f"d_rel_{c}"] = float(stack(v, f"rel_{c}").mean() - stack("B", f"rel_{c}").mean())
        dec.append(row)
    write_csv(args.out_dir / "a1_decomposition.csv", dec)
    md += ["## A1 Channel weights in rel_all", "",
           f"Identity rel_all_s = sum_c w_cs rel_cs, w_cs = sum|t_c| / sum_c' sum|t_c'|: max |lhs - rhs| = {ident:.3g}.", "",
           md_table(a1, ["channel", "w_mean", "w_min", "w_max"]), "",
           "Decomposition of the difference of mean rel_all (seed means; percentage points; contributions sum "
           "to d_rel_all):", "",
           md_table(dec, ["comparison", "d_rel_all", "contrib_p", "contrib_vx", "contrib_vy", "d_rel_p", "d_rel_vx",
                          "d_rel_vy"], {k: pct for k in ["d_rel_all", "contrib_p", "contrib_vx", "contrib_vy",
                                                         "d_rel_p", "d_rel_vx", "d_rel_vy"]}), ""]

    # --- A2: MAE per channel, fluid and interior ---
    a2 = []
    tgt_fluid = None
    for label in labels:
        row = {"config": label, "seeds": len(res[label])}
        for where, regs in (("fluid", ("wall", "wake", "rest")), ("interior", ("interior",))):
            err = sum(stack(label, f"reg_err_{r}") for r in regs)  # (S, N, 3)
            cnt = sum(stack(label, f"reg_n_{r}") for r in regs)  # (S, N)
            mae = (err / cnt[..., None]).mean(axis=1).mean(axis=0)  # mean over samples, then seeds
            for i, c in enumerate(CH):
                row[f"mae_{where}_{c}"] = mae[i]
        a2.append(row)
        if tgt_fluid is None:
            t = sum(stack(label, f"reg_tgt_{r}")[0] for r in ("wall", "wake", "rest"))
            cnt = sum(stack(label, f"reg_n_{r}")[0] for r in ("wall", "wake", "rest"))
            tgt_fluid = (t / cnt[:, None]).mean(axis=0)
    tgt_int = (stack(labels[0], "reg_tgt_interior")[0] / stack(labels[0], "reg_n_interior")[0][:, None]).mean(0)
    a2.append({"config": "mean |target|", "seeds": "",
               **{f"mae_fluid_{c}": tgt_fluid[i] for i, c in enumerate(CH)},
               **{f"mae_interior_{c}": tgt_int[i] for i, c in enumerate(CH)}})
    write_csv(args.out_dir / "a2_mae.csv", a2)
    e3 = lambda v: f"{1e3 * v:.4f}" if isinstance(v, float) else str(v)
    md += ["## A2 Mean absolute error per channel (normalised fields, x1e-3)", "",
           "Per sample: mean |o - t| over the cells of the region; then mean over samples and seeds. The last row is "
           "the mean |target| over the same cells (scale of each channel; computed, 0 inside the airfoil).", "",
           md_table(a2, ["config", "seeds"] + [f"mae_fluid_{c}" for c in CH] + [f"mae_interior_{c}" for c in CH],
                    {f"mae_{w_}_{c}": e3 for w_ in ("fluid", "interior") for c in CH}), ""]

    # --- A3: paired comparisons ---
    a3 = []
    for v, ref in COMPARISONS:
        if v not in res or ref not in res:
            continue
        for m in METRICS:
            b = explore.paired_bootstrap(stack(v, m), stack(ref, m), profiles, args.n_boot, args.boot_seed)
            a3.append({"comparison": f"{v} - {ref}", "metric": m, **b})
    write_csv(args.out_dir / "a3_paired.csv", a3)
    ci = lambda r, k: f"[{100 * r[k + '_lo']:+.3f}, {100 * r[k + '_hi']:+.3f}]"
    rows_md = [{"comparison": r["comparison"], "metric": r["metric"], "seeds": f"{r['seeds_var']}/{r['seeds_ref']}",
                "mean diff": f"{100 * r['mean_diff']:+.3f}", "95% CI profiles": ci(r, "ci1"),
                "95% CI seeds+profiles": ci(r, "ci2"), "median diff": f"{100 * r['median_diff']:+.3f}",
                "profiles better": f"{r['frac_profiles_var_better']:.2f}",
                "samples better": f"{r['frac_samples_var_better']:.2f}"} for r in a3]
    md += ["## A3 Paired per-sample comparisons (percentage points; negative = variant lower error)", "",
           f"{n} samples, {len(set(profiles))} profiles. CI one level: profiles resampled; two levels: seeds of each "
           "configuration and profiles resampled (with 3 seeds the seed level is coarse).", "",
           md_table(rows_md, list(rows_md[0].keys())), ""]

    # --- A4: strata ---
    crit = {"abs_alpha": (alpha, "|alpha| (deg)"), "p_signal": (psum, "sum |p target| (normalised)"),
            "thickness": (thick, "relative thickness (mask)")}
    ext = [e for e in args.external.split(",") if e in res]
    ext_seeds = parse_seeds(args.external_seeds)
    if ext:
        def ext_rel_p(e):
            pick = [j for j, (s_, _, _) in enumerate(runs[e]) if ext_seeds is None or s_ in ext_seeds]
            return stack(e, "rel_p")[pick].mean(0)
        crit["external"] = (np.mean([ext_rel_p(e) for e in ext], axis=0),
                            f"external difficulty: seed-mean rel_p of {'+'.join(ext)}"
                            + (f" (seeds {args.external_seeds})" if ext_seeds else ""))
    names = list(crit)
    ranks = {c: np.argsort(np.argsort(crit[c][0])) for c in names}
    spear = [{"criterion": a, **{b: float(np.corrcoef(ranks[a], ranks[b])[0, 1]) for b in names}} for a in names]
    write_csv(args.out_dir / "a4_criteria_spearman.csv", spear)
    a4 = []
    for cname, (vals, desc) in crit.items():
        edges = np.quantile(vals, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        q = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, 4)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, m in zip(axes, ["rel_all", "rel_p"]):
            for k, (v, ref) in enumerate(STRATA_COMPARISONS):
                if v not in res or ref not in res:
                    continue
                pts = []
                for qi in range(5):
                    sub = q == qi
                    b = explore.paired_bootstrap(stack(v, m), stack(ref, m), profiles, args.n_boot,
                                                 args.boot_seed, two_level=True, subset=sub)
                    a4.append({"criterion": cname, "quintile": qi + 1, "lo": edges[qi], "hi": edges[qi + 1],
                               "comparison": f"{v} - {ref}", "metric": m, "n_samples": b["n_samples"],
                               "n_profiles": b["n_profiles"], "mean_diff": b["mean_diff"], "ci_lo": b["ci1_lo"],
                               "ci_hi": b["ci1_hi"], "ci2_lo": b["ci2_lo"], "ci2_hi": b["ci2_hi"], "ref_mean": float(stack(ref, m).mean(0)[sub].mean())})
                    pts.append((b["mean_diff"], b["ci1_lo"], b["ci1_hi"]))
                pts = 100 * np.array(pts)
                x = np.arange(1, 6) + (k - 2) * 0.08
                ax.errorbar(x, pts[:, 0], yerr=[pts[:, 0] - pts[:, 1], pts[:, 2] - pts[:, 0]], fmt="o-", ms=4,
                            capsize=2, label=f"{v} - {ref}")
            ax.axhline(0, color="0.5", lw=0.8)
            ax.set_xticks(range(1, 6), [f"Q{i}\n{edges[i - 1]:.3g}-{edges[i]:.3g}" for i in range(1, 6)], fontsize=7)
            ax.set_ylabel(f"mean difference of {m} (points)")
            ax.set_title(m)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"A4 (exploratory) differences by quintile of {desc}; 95% CI bootstrap over profiles",
                     fontsize=9)
        fig.savefig(args.fig_dir / f"a4_{cname}.png", dpi=120)
        plt.close(fig)
    write_csv(args.out_dir / "a4_strata.csv", a4)
    md += ["## A4 Differences by quintile (percentage points, 95% CI over profiles)", "",
           "Criteria: |alpha| = |atan2(vy_inf, vx_inf)|; sum |p target|; relative thickness from the mask; external "
           f"difficulty = seed-mean rel_p of {'+'.join(ext) or 'n/a'}"
           + (f" (seeds {args.external_seeds})" if ext_seeds else "") + ". Cells: mean diff [one-level CI]; the "
           "two-level CI (seeds + profiles) is in a4_strata.csv (ci2_lo, ci2_hi). Figures: "
           f"`{args.fig_dir.relative_to(REPO) if args.fig_dir.is_relative_to(REPO) else args.fig_dir}/a4_<criterion>.png`.",
           ""]
    md += ["Spearman correlation between criteria:", "",
           md_table(spear, ["criterion"] + names, {b: (lambda v: f"{v:+.2f}") for b in names}), ""]
    for cname in crit:
        sub = [r for r in a4 if r["criterion"] == cname]
        tab = []
        for qi in range(1, 6):
            rq = [r for r in sub if r["quintile"] == qi]
            row = {"Q": qi, "range": f"{rq[0]['lo']:.3g}-{rq[0]['hi']:.3g}", "n/prof": f"{rq[0]['n_samples']}/{rq[0]['n_profiles']}"}
            for r in rq:
                row[f"{r['comparison']} {r['metric'][4:]}"] = (f"{100 * r['mean_diff']:+.2f} "
                                                               f"[{100 * r['ci_lo']:+.2f}, {100 * r['ci_hi']:+.2f}]")
            tab.append(row)
        md += [f"### {cname}", "", md_table(tab, list(tab[0].keys())), ""]

    # --- A5: maps and regions ---
    maps = {label: stack(label, "mean_abs_err_map").mean(0) for label in labels}
    mean_mask = ds.inputs[:, 2].numpy().mean(0)
    show = [l for l in ("B", "MD8", "SA8") if l in maps]
    cols = show + [f"{l} - B" for l in show if l != "B"]
    fig, axes = plt.subplots(3, len(cols), figsize=(2.6 * len(cols), 7.5), constrained_layout=True)
    for i, c in enumerate(CH):
        vmax = max(maps[l][i].max() for l in show)
        dmax = max([np.abs(maps[l][i] - maps["B"][i]).max() for l in show if l != "B"] or [1])
        for j, col in enumerate(cols):
            ax = axes[i, j]
            if " - " in col:
                img = maps[col.split(" - ")[0]][i] - maps["B"][i]
                im = ax.imshow(img, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
            else:
                im = ax.imshow(maps[col][i], cmap="viridis", vmin=0, vmax=vmax)
            ax.contour(mean_mask, levels=[0.5], colors="w", linewidths=0.5)
            ax.set_title(f"{col}: |err| {c}", fontsize=8)
            ax.set_xticks([]), ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("A5 (exploratory) mean |error| over samples and seeds (normalised fields); rows = x (flow "
                 "downwards), columns = y; white: mean airfoil mask", fontsize=9)
    fig.savefig(args.fig_dir / "a5_maps.png", dpi=120)
    plt.close(fig)

    a5, a5d = [], []
    reg_mae = {}
    for label in labels:
        for r in explore.REGIONS:
            e = stack(label, f"reg_err_{r}") / np.maximum(stack(label, f"reg_n_{r}"), 1)[..., None]  # (S, N, 3)
            reg_mae[(label, r)] = e
            m = e.mean(axis=(0, 1))
            a5.append({"config": label, "region": r, **{f"mae_{c}": m[i] for i, c in enumerate(CH)}})
    cells = {r: np.mean([reg[r].sum() for reg in regions]) for r in explore.REGIONS}
    for v, ref in [c for c in COMPARISONS if c[1] == "B" or c == ("MD8", "SA8")]:
        if v not in res or ref not in res:
            continue
        for r in ("wall", "wake", "rest"):
            for i, c in enumerate(CH):
                b = explore.paired_bootstrap(reg_mae[(v, r)][..., i], reg_mae[(ref, r)][..., i], profiles,
                                             args.n_boot, args.boot_seed, two_level=False)
                a5d.append({"comparison": f"{v} - {ref}", "region": r, "channel": c, "mean_diff": b["mean_diff"],
                            "ci_lo": b["ci1_lo"], "ci_hi": b["ci1_hi"],
                            "rel_change": b["mean_diff"] / reg_mae[(ref, r)][..., i].mean()})
    write_csv(args.out_dir / "a5_regions.csv", a5)
    write_csv(args.out_dir / "a5_region_diffs.csv", a5d)
    md += ["## A5 Errors by region (MAE x1e-3, normalised fields)", "",
           f"Regions (mdf/explore.py): wall = fluid cells 1-{explore.WALL_PX} px from the airfoil; wake = fluid "
           f"cells downstream of the trailing edge within {explore.WAKE_HALF_WIDTH} px of the free-stream line; "
           "rest = other fluid cells; interior = airfoil cells (target 0, control). Mean cells per sample: "
           + ", ".join(f"{r} {cells[r]:.0f}" for r in explore.REGIONS) + ". Maps: `a5_maps.png`.", "",
           md_table(a5, ["config", "region"] + [f"mae_{c}" for c in CH], {f"mae_{c}": e3 for c in CH}), "",
           "Differences (x1e-3, 95% CI over profiles; rel_change = mean diff / reference MAE):", "",
           md_table([{"comparison": r["comparison"], "region": r["region"], "channel": r["channel"],
                      "diff": f"{1e3 * r['mean_diff']:+.4f} [{1e3 * r['ci_lo']:+.4f}, {1e3 * r['ci_hi']:+.4f}]",
                      "rel_change": f"{100 * r['rel_change']:+.1f} %"} for r in a5d],
                    ["comparison", "region", "channel", "diff", "rel_change"]), ""]

    # --- A6: train / val gap ---
    a6, curves = [], {}
    for label, lst in runs.items():
        for seed, d, cfg in lst:
            with open(d / "metrics.csv") as fh:
                rows = list(csv.DictReader(fh))
            tr = np.array([float(r["train_L1_mean_of_batches"]) for r in rows])
            va = np.array([float(r["val_L1_mean_of_batches"]) for r in rows])
            curves.setdefault(label, []).append((tr, va))
            a6.append({"config": label, "seed": seed, "train_L1_last": tr[-1], "val_L1_last": va[-1],
                       "gap": va[-1] - tr[-1], "ratio": va[-1] / tr[-1], "val_L1_min": va.min(),
                       "epoch_val_min": int(va.argmin())})
    write_csv(args.out_dir / "a6_gap.csv", a6)
    a6m = []
    for label in labels:
        rr = [r for r in a6 if r["config"] == label]
        a6m.append({"config": label, "seeds": len(rr),
                    **{k: float(np.mean([r[k] for r in rr])) for k in ("train_L1_last", "val_L1_last", "gap", "ratio")},
                    "gap_sd": float(np.std([r["gap"] for r in rr], ddof=1)) if len(rr) > 1 else float("nan")})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for label in labels:
        tr = np.mean([c[0] for c in curves[label]], axis=0)
        va = np.mean([c[1] for c in curves[label]], axis=0)
        axes[0].plot(va, lw=1, label=label)
        axes[1].plot(np.arange(len(va))[-40:], (va - tr)[-40:], lw=1, label=label)
    axes[0].set_yscale("log"), axes[0].set_xlabel("epoch"), axes[0].set_ylabel("val L1 (mean over seeds)")
    axes[1].set_xlabel("epoch"), axes[1].set_ylabel("val L1 - train L1 (last 40 epochs)")
    axes[0].legend(fontsize=7)
    fig.suptitle("A6 (exploratory) validation curves and train/validation gap (metrics.csv)", fontsize=9)
    fig.savefig(args.fig_dir / "a6_curves.png", dpi=120)
    plt.close(fig)
    md += ["## A6 Train / validation L1 at the last epoch (x1e-3, mean over seeds)", "",
           "train L1 = mean over the batches of the last epoch (weights changing, train mode; dropout 0); "
           "val L1 = 40 batches of 10, eval mode. Figure `a6_curves.png`.", "",
           md_table(a6m, ["config", "seeds", "train_L1_last", "val_L1_last", "gap", "gap_sd", "ratio"],
                    {k: e3 for k in ("train_L1_last", "val_L1_last", "gap", "gap_sd")}), ""]

    # --- A7: qualitative ---
    a7 = []
    q_labels = [l for l in ("B", "MD8", "SA8") if l in res]
    preds = {}
    for l in q_labels:
        pos = [s for s, _, _ in runs[l]]
        if args.qual_seed not in pos:
            sys.exit(f"{l}: seed {args.qual_seed} not available for A7")
        preds[l] = res[l][pos.index(args.qual_seed)]["keep_pred"]
    for k, i in enumerate(keep):
        f = ds.files[i]
        tgt = ds.targets[i].numpy()
        mask = ds.inputs[i, 2].numpy() > 0.5
        kind = "high |alpha|" if i in hi_alpha else "small p signal"
        fig, axes = plt.subplots(2, 1 + 2 * len(q_labels), figsize=(2.3 * (1 + 2 * len(q_labels)), 5),
                                 constrained_layout=True)
        for row, (ci_, c) in enumerate([(0, "p"), (2, "vy")]):
            lim = np.abs(tgt[ci_]).max()
            elim = max(np.abs(preds[l][k][ci_] - tgt[ci_]).max() for l in q_labels)
            panels = [("target", tgt[ci_], "RdBu_r", lim)]
            panels += [(f"{l} pred", preds[l][k][ci_], "RdBu_r", lim) for l in q_labels]
            panels += [(f"{l} err", preds[l][k][ci_] - tgt[ci_], "PuOr_r", elim) for l in q_labels]
            for j, (title, img, cmap, vlim) in enumerate(panels):
                ax = axes[row, j]
                im = ax.imshow(np.where(mask, np.nan, img), cmap=cmap, vmin=-vlim, vmax=vlim)
                ax.set_title(f"{c} {title}", fontsize=7)
                ax.set_xticks([]), ax.set_yticks([])
            fig.colorbar(im, ax=axes[row, -1], shrink=0.7)
        fig.suptitle(f"A7 (exploratory) {f} ({kind}, alpha {props[i]['alpha_deg']:.1f} deg, sum|p| "
                     f"{props[i]['sum_abs_p_tgt']:.1f}); seed {args.qual_seed}; flow downwards", fontsize=8)
        fig.savefig(args.fig_dir / f"a7_{Path(f).stem}.png", dpi=110)
        plt.close(fig)
        # separation indicators: reverse flow (vx < 0) in the fluid downstream of the trailing edge
        te_r, _ = explore.trailing_edge(mask)
        down = (~mask) & (np.arange(128)[:, None] > te_r)
        near = (~mask) & (explore.distance_to_airfoil(mask) <= 8)
        row = {"file": f, "kind": kind, "alpha_deg": props[i]["alpha_deg"], "sum_abs_p_tgt": props[i]["sum_abs_p_tgt"],
               "tgt_vx_neg_cells_downstream": int((tgt[1][down] < 0).sum()),
               "tgt_vx_neg_cells_near_wall": int((tgt[1][near] < 0).sum())}
        for l in q_labels:
            row[f"{l}_vx_neg_cells_downstream"] = int((preds[l][k][1][down] < 0).sum())
            row[f"{l}_rel_p"] = float(res[l][[s for s, _, _ in runs[l]].index(args.qual_seed)]["rel_p"][i])
            row[f"{l}_rel_vy"] = float(res[l][[s for s, _, _ in runs[l]].index(args.qual_seed)]["rel_vy"][i])
        a7.append(row)
    # flow view: target vx with reverse-flow cells, high-alpha cases + the lowest-|alpha| sample as reference
    ref_i = int(np.argmin(alpha))
    show_i = hi_alpha + [ref_i]
    fig, axes = plt.subplots(2, len(show_i), figsize=(3.2 * len(show_i), 6.5), constrained_layout=True)
    for j, i in enumerate(show_i):
        tgt = ds.targets[i].numpy()
        mask = ds.inputs[i, 2].numpy() > 0.5
        vx = np.where(mask, np.nan, tgt[1])
        ax = axes[0, j]
        im = ax.imshow(vx, cmap="viridis")
        ax.contour(np.where(mask, 1.0, tgt[1]), levels=[0.0], colors="r", linewidths=0.8)
        ax.set_title(f"{Path(ds.files[i]).stem}\nalpha {props[i]['alpha_deg']:.1f}: target vx, red: vx=0", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax = axes[1, j]
        prof_rows = [r for r in (60, 80, 100, 120) if r < 128]
        for r in prof_rows:
            ax.plot(tgt[1][r], label=f"row {r}")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title("target vx across y at fixed x rows", fontsize=7)
        ax.legend(fontsize=6)
        for a in axes[:, j]:
            a.tick_params(labelsize=6)
    fig.suptitle("A7 (exploratory) is the flow separated? Reverse flow (vx < 0) and wake width; flow downwards",
                 fontsize=9)
    fig.savefig(args.fig_dir / "a7_flow_high_alpha.png", dpi=110)
    plt.close(fig)
    write_csv(args.out_dir / "a7_samples.csv", a7)
    md += ["## A7 Qualitative samples (seed %d)" % args.qual_seed, "",
           "Selection from data only: the --qual-files hard cases present in the split, filled up to 3 by largest "
           "|alpha|; 3 smallest sum |p target|. Figures `a7_<file>.png`, "
           "`a7_flow_high_alpha.png`. vx_neg cells: fluid cells with vx < 0 downstream of the trailing edge "
           "(reverse flow), target and predictions; near_wall: within 8 px of the airfoil.", "",
           md_table(a7, list(a7[0].keys())), ""]

    if args.previous is not None:
        md += earlier_vs_now(args.previous, a3, a4, args.out_dir)

    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {args.out_dir}/summary.md and figures in {args.fig_dir}")


if __name__ == "__main__":
    main()
