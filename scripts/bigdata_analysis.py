"""Pre-registered analysis of the big-dataset experiment (docs/preregistration-bigdata.md; exploratory, validation only).

Runs on the laptop after scripts/cloud/sync_results.sh; needs no data set: per-sample values come from each run's
eval_val_samples.csv (evaluate.py) and, for the full set, post_val.npz / post_trainfit.json (scripts/cloud/post_bigdata.py).
The test set is never read.

Sets
  full     runs <full-root>/<label>_s<seed> (queues/bigdata.txt), split mode grouped-full: validation = all files of the 85
           grouped validation profiles in the used subdirectories; profiles from mdata.profile_full
  reduced  the existing 6.4k runs with the same seeds (default: B night1/ep133_s*, MDv64-local night8a, MDv64 night5),
           split mode grouped (402 files, the same 85 profiles); profiles from mdata.profile
Validation subsets of the full set (Amendment 1 of the pre-registration, 2026-10-19): full-reg = the reg/ files of the
85 profiles, the MAIN set (the reduced-set validation was all reg/); full-shear = the shear/ files and full-all = all
files, both descriptive. The subdirectory is taken from each file name in eval_val_samples.csv ("<subdir>/<name>.npz").
Comparisons (reduced and each full subset, the seeds common to the two labels of a pair; missing runs declared):
  MDv64-local - B, MDv64 - B, MDv64-local - MDv64
  main: rel_vy (mean of per-sample ratios, original definition); descriptive: aggregate and median rel_vy, rel_all,
  rel_p; full set only: wall-band vy MAE (x1e-3), divergence error wall / other (x1e-3), verification 5a (train and
  val L1 in eval mode, (mean_var - mean_B) / SE_Welch).
  Paired differences of per-sample seed means; 95 % intervals from the two-level bootstrap (seeds of each configuration
  + profiles; explore.paired_bootstrap / paired_bootstrap_agg_median), 10 000 repetitions, seed 0, as all earlier
  analyses. Reading: qualitative comparison of the full-set effect with the reduced-set one; no confirmatory decision.
Outputs: <out-dir>/summary.md, comparisons.csv, runs.csv; <fig-dir>/bigdata_rel_vy.{pdf,png} (IEEE one column: rel_vy
of B and MDv64-local per seed and mean, reduced set vs the full-reg validation side by side).

Usage:
    python scripts/bigdata_analysis.py --full-root ~/runs/bigdata-vm/runs/bigdata [--seeds 0-2]
        [--reduced LABEL=GLOB ...] [--out-dir docs/results/val-bigdata] [--fig-dir docs/figures/val-bigdata]
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `mdf`
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/, for analyze_val helpers

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_val as av
from mdf import data as mdata
from mdf import explore, runinfo

LABELS = ("B", "MDv64-local", "MDv64")
PAIRS = [("MDv64-local", "B"), ("MDv64", "B"), ("MDv64-local", "MDv64")]
FULL_DIRS = {"B": "B_s*", "MDv64-local": "MDv64-local_s*", "MDv64": "MDv64_s*"}
REDUCED_DEFAULT = ["B=~/runs/night1/ep133_s*", "MDv64-local=~/runs/night8a/MDv64-local_s*",
                   "MDv64=~/runs/night5/MDv64_s*"]
N_BOOT, BOOT_SEED = 10000, 0
SET_NAMES = {"reduced": "reduced (6.4k), validation 402 reg", "full-reg": "full (53.8k), validation reg/ (MAIN)",
             "full-shear": "full (53.8k), validation shear/ (descriptive)",
             "full-all": "full (53.8k), whole validation (descriptive)"}
FULL_SUBSETS = {"full-reg": "reg/", "full-shear": "shear/", "full-all": ""}  # file-name prefix


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full-root", type=Path, required=True)
    ap.add_argument("--reduced", action="append", default=None, help="LABEL=GLOB of reduced-set runs (repeatable)")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--out-dir", type=Path, default=Path("docs/results/val-bigdata"))
    ap.add_argument("--fig-dir", type=Path, default=Path("docs/figures/val-bigdata"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    return ap.parse_args()


def read_samples(run_dir):
    with open(run_dir / "eval_val_samples.csv") as fh:
        rows = list(csv.DictReader(fh))
    out = {"file": [r["file"] for r in rows]}
    for k in ("rel_vy", "rel_all", "rel_p", "abs_err_vy", "abs_tgt_vy", "L1"):
        out[k] = np.array([float(r[k]) for r in rows])
    return out


def collect(specs, seeds, expect_mode):
    """label -> list of (seed, run_dir, cfg, samples); every run finished, evaluated, of the expected split mode."""
    runs = {}
    for spec in specs:
        label, _, pattern = spec.partition("=")
        for d in sorted(glob.glob(os.path.expanduser(pattern))):
            d = Path(d)
            if not (d / "config.json").exists() or not (d / "eval_val_samples.csv").exists():
                continue
            cfg = json.loads((d / "config.json").read_text())
            if cfg.get("status") != "finished" or cfg["seed"] not in seeds:
                continue
            mode = cfg.get("split", {}).get("split_mode")
            if mode != expect_mode:
                sys.exit(f"{d}: split mode {mode}, expected {expect_mode}")
            runs.setdefault(label, []).append((cfg["seed"], d, cfg, read_samples(d)))
    for label in runs:
        runs[label].sort(key=lambda t: t[0])
        if len({t[0] for t in runs[label]}) != len(runs[label]):
            sys.exit(f"duplicate seeds for {label}: {[t[0] for t in runs[label]]}")
    return runs


def check_same_files(runs, name):
    files = {tuple(t[3]["file"]) for lst in runs.values() for t in lst}
    shas = {t[2]["split"]["val_files_sha256"] for lst in runs.values() for t in lst}
    if len(files) != 1 or len(shas) != 1:
        sys.exit(f"{name}: runs differ in validation files ({len(files)} lists, {len(shas)} hashes)")
    files = list(next(iter(files)))
    if mdata.file_list_sha256(files) not in shas:
        sys.exit(f"{name}: eval CSV file list does not match the recorded val_files_sha256")
    return files


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size) if min(a.size, b.size) > 1 else float("nan")
    d = a.mean() - b.mean()
    return d, se, d / se if se else float("nan")


def fmt_ci(bs, scale):
    return (f"{scale * bs['mean_diff']:+.3f} [{scale * bs['ci2_lo']:+.3f}, {scale * bs['ci2_hi']:+.3f}]",
            scale * bs["mean_diff"], scale * bs["ci2_lo"], scale * bs["ci2_hi"])


def compare(set_name, runs, profiles, post, args):
    rows = []
    for v, ref in PAIRS:
        if v not in runs or ref not in runs:
            rows.append({"set": set_name, "comparison": f"{v} - {ref}", "metric": "all", "seeds": "",
                         "estimate [95% CI]": "not computed: runs missing", "diff": "", "lo": "", "hi": ""})
            continue
        common = sorted({t[0] for t in runs[v]} & {t[0] for t in runs[ref]})
        sel = {lab: [t for t in runs[lab] if t[0] in common] for lab in (v, ref)}

        def st(lab, key):
            return np.stack([t[3][key] for t in sel[lab]])

        def add(metric, bs, scale):
            text, d, lo, hi = fmt_ci(bs, scale)
            rows.append({"set": set_name, "comparison": f"{v} - {ref}", "metric": metric,
                         "seeds": ",".join(map(str, common)), "estimate [95% CI]": text, "diff": d, "lo": lo, "hi": hi})

        for m in ("rel_vy", "rel_all", "rel_p"):
            add(m, explore.paired_bootstrap(st(v, m), st(ref, m), profiles, args.n_boot, args.boot_seed), 100)
        tgt = st(v, "abs_tgt_vy")[0]
        am = explore.paired_bootstrap_agg_median(st(v, "abs_err_vy"), st(ref, "abs_err_vy"), tgt, st(v, "rel_vy"),
                                                 st(ref, "rel_vy"), profiles, args.n_boot, args.boot_seed)
        for kind in ("agg", "median"):
            add(f"{kind}_rel_vy", {"mean_diff": am[f"{kind}_diff"], "ci2_lo": am[f"{kind}_ci2_lo"],
                                   "ci2_hi": am[f"{kind}_ci2_hi"]}, 100)
        if post is not None:
            def pst(lab, key):
                return np.stack([post[(lab, t[0])][key] for t in sel[lab]])
            wall = {lab: pst(lab, "reg_err_wall")[..., 2] / np.maximum(pst(lab, "reg_n_wall"), 1) for lab in (v, ref)}
            add("wall_vy_mae_x1e-3", explore.paired_bootstrap(wall[v], wall[ref], profiles, args.n_boot,
                                                              args.boot_seed), 1e3)
            for g in explore.DIV_REGIONS:
                add(f"div_err_{g}_x1e-3", explore.paired_bootstrap(pst(v, f"div_err_{g}"), pst(ref, f"div_err_{g}"),
                                                                   profiles, args.n_boot, args.boot_seed), 1e3)
    return rows


def load_post(runs, files):
    """(label, seed) -> post_val.npz arrays, or None if any run lacks it; checks the file order and rel_vy."""
    post, max_diff = {}, 0.0
    for label, lst in runs.items():
        for seed, d, _, s in lst:
            p = d / "post_val.npz"
            if not p.exists():
                return None, None
            with np.load(p) as z:
                arr = {k: z[k] for k in z.files}
            if list(arr["files"]) != files:
                sys.exit(f"{p}: file order differs from eval_val_samples.csv")
            max_diff = max(max_diff, float(np.max(np.abs(arr["rel_vy"] - s["rel_vy"]))))
            post[(label, seed)] = arr
    return post, max_diff


def subset(runs, post, prefix):
    """Runs and post-processing arrays restricted to the validation files whose name starts with prefix."""
    files = runs[next(iter(runs))][0][3]["file"]
    mask = np.array([f.startswith(prefix) for f in files])
    n = mask.size
    out = {}
    for label, lst in runs.items():
        out[label] = []
        for seed, d, cfg, s in lst:
            s2 = {k: ([f for f, m in zip(v, mask) if m] if k == "file" else v[mask]) for k, v in s.items()}
            out[label].append((seed, d, cfg, s2))
    post2 = None
    if post is not None:
        post2 = {key: {k: (v[mask] if getattr(v, "ndim", 0) >= 1 and v.shape[0] == n else v) for k, v in arr.items()}
                 for key, arr in post.items()}
    return out, post2, int(mask.sum())


def figure(path_base, rel_vy_means):
    """IEEE one column: rel_vy per seed (small markers) and mean (large) of B and MDv64-local, reduced vs full."""
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)
    style = {"B": ("o", "0.55", -0.13), "MDv64-local": ("s", "0.0", 0.13)}
    for j, set_name in enumerate(("reduced", "full-reg")):
        for label, (mk, col, off) in style.items():
            vals = rel_vy_means.get((set_name, label))
            if vals is None:
                continue
            x = j + off
            jit = (np.arange(vals.size) - (vals.size - 1) / 2) * 0.03
            ax.scatter(x + jit, vals, s=10, marker=mk, facecolors="none", edgecolors=col, linewidths=0.7, zorder=2)
            ax.scatter([x], [vals.mean()], s=110, marker="_", color=col, linewidths=1.8, zorder=3)
            ax.annotate(f"{vals.mean():.2f}", (x, vals.mean()), xytext=(13, 0), textcoords="offset points",
                        va="center", fontsize=6, color="0.25")
    for label, (mk, col, _) in style.items():
        ax.scatter([], [], marker=mk, facecolors="none", edgecolors=col, s=14, label=f"{label}: seeds, bar = mean")
    n = {s: next((v.size for (ss, _), v in rel_vy_means.items() if ss == s), 0) for s in ("reduced", "full-reg")}
    ax.set_xticks([0, 1], [f"trained on 6.4k\n{n['reduced']} seeds",
                           f"trained on 53.8k\n{n['full-reg']} seeds"])
    ax.set_title("validation: regular airfoils of the 85 validation profiles", fontsize=6.5, loc="center")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("validation rel. error $v_y$ (%)")
    ax.grid(axis="y", color="0.92", lw=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.3,
              columnspacing=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_base}.{ext}", dpi=300)
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


def main():
    args = parse_args()
    seeds = av.parse_seeds(args.seeds)
    full = collect([f"{lab}={args.full_root}/{pat}" for lab, pat in FULL_DIRS.items()], seeds, mdata.GROUPED_FULL)
    reduced = collect(args.reduced or REDUCED_DEFAULT, seeds, "grouped")
    if not full:
        sys.exit(f"no finished, evaluated grouped-full runs under {args.full_root}")
    files = {"full": check_same_files(full, "full"), "reduced": check_same_files(reduced, "reduced")}
    profiles = {"full": [mdata.profile_full(f) for f in files["full"]],
                "reduced": [mdata.profile(f) for f in files["reduced"]]}
    if set(profiles["full"]) != set(profiles["reduced"]):
        print("WARNING: the full and reduced validation sets do not have the same profiles")
    post, post_diff = load_post(full, files["full"])
    sets, posts, profs, n_sub = {"reduced": reduced}, {"reduced": None}, {"reduced": profiles["reduced"]}, {}
    for sname, prefix in FULL_SUBSETS.items():
        sets[sname], posts[sname], n_sub[sname] = subset(full, post, prefix)
        if n_sub[sname] == 0:
            sys.exit(f"validation subset {sname} is empty (file names without '{prefix}'?)")
        profs[sname] = [mdata.profile_full(f) for f in sets[sname][next(iter(full))][0][3]["file"]]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    run_rows, rel_vy_means = [], {}
    for sname, runs in sets.items():
        for label in LABELS:
            for seed, d, cfg, s in runs.get(label, []):
                run_rows.append({"set": sname, "label": label, "seed": seed,
                                 "run_dir": str(d).replace(str(Path.home()), "~"),
                                 "commit": cfg.get("git", {}).get("commit", "")[:7], "n_val": len(s["file"]),
                                 "n_train": cfg["split"]["n_train"], "rel_vy_pct": 100 * s["rel_vy"].mean(),
                                 "rel_all_pct": 100 * s["rel_all"].mean(), "it_per_s": cfg.get("it_per_s_train")})
            if label in runs:
                rel_vy_means[(sname, label)] = np.array([100 * t[3]["rel_vy"].mean() for t in runs[label]])
    av.write_csv(args.out_dir / "runs.csv", run_rows)

    rows = [r for sname in sets for r in compare(sname, sets[sname], profs[sname], posts[sname], args)]
    av.write_csv(args.out_dir / "comparisons.csv", rows)

    fit_rows = []
    if post is not None and all((t[1] / "post_trainfit.json").exists() for lst in full.values() for t in lst):
        fit = {lab: [json.loads((t[1] / "post_trainfit.json").read_text()) for t in full[lab]] for lab in full}
        for v in ("MDv64-local", "MDv64"):
            if v in fit and "B" in fit:
                for k in ("train_L1", "train_L1_reg", "train_L1_shear", "val_L1", "val_L1_reg", "val_L1_shear"):
                    if not all(k in f for f in fit[v] + fit["B"]):
                        continue
                    d, se, z = welch([f[k] for f in fit[v]], [f[k] for f in fit["B"]])
                    fit_rows.append({"comparison": f"{v} - B", "quantity": k, "diff": d, "se_welch": se,
                                     "diff_over_se": z})
        if fit_rows:
            av.write_csv(args.out_dir / "trainfit_5a.csv", fit_rows)
    figure(args.fig_dir / "bigdata_rel_vy", rel_vy_means)

    missing = {s: {lab: sorted(seeds - {t[0] for t in runs.get(lab, [])}) for lab in LABELS} for s, runs in sets.items()}
    md = ["# Big-dataset experiment: validation analysis (exploratory; docs/preregistration-bigdata.md)", "",
          f"Generated by `scripts/bigdata_analysis.py`, git {runinfo.git_info()}, {runinfo.utc_now()}. "
          f"Validation only (the test set is never read). Bootstrap two-level, {args.n_boot} repetitions, seed "
          f"{args.boot_seed}. Differences in percentage points unless stated. No confirmatory decision.", "",
          f"- Full set: {len(files['full'])} validation samples, {len(set(profiles['full']))} profiles: "
          + ", ".join(f"{s} {n_sub[s]} samples / {len(set(profs[s]))} profiles" for s in FULL_SUBSETS)
          + f"; reduced set: {len(files['reduced'])} samples, {len(set(profiles['reduced']))} profiles.",
          "- MAIN validation of the full set: full-reg (Amendment 1); full-shear and full-all are descriptive.",
          f"- Seeds requested {sorted(seeds)}; missing (declared): " + "; ".join(
              f"{s}: " + ", ".join(f"{lab} {m}" for lab, m in mm.items() if m) for s, mm in missing.items()
              if any(mm.values())) if any(any(mm.values()) for mm in missing.values()) else
          f"- Seeds requested {sorted(seeds)}; none missing.",
          f"- post_val.npz: {'present, max |rel_vy - eval CSV| ' + format(post_diff, '.2g') if post else 'missing'}.",
          "", "## Mean rel_vy per configuration (%, mean (std) over seeds)", "",
          "| set | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    for sname in sets:
        cells = []
        for lab in LABELS:
            v = rel_vy_means.get((sname, lab))
            cells.append("-" if v is None else f"{v.mean():.3f} ({v.std(ddof=1) if v.size > 1 else 0:.3f}), n={v.size}")
        md.append(f"| {SET_NAMES[sname]} | " + " | ".join(cells) + " |")
    md += ["", "## Comparisons (two-level 95 % CI)", "", av.md_table(rows, ["set", "comparison", "metric", "seeds",
                                                                           "estimate [95% CI]"]), ""]
    if fit_rows:
        md += ["## Verification 5a (full set; L1 in eval mode, (mean_var - mean_B) / SE_Welch)", "",
               av.md_table([{k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()} for r in fit_rows],
                           list(fit_rows[0])), ""]
    main_rows = {(r["set"], r["comparison"]): r for r in rows if r["metric"] == "rel_vy"}
    md += ["## Reading (pre-registered: qualitative; main = full-reg)", ""]
    for v, ref in PAIRS:
        a, b = main_rows.get(("reduced", f"{v} - {ref}")), main_rows.get(("full-reg", f"{v} - {ref}"))
        if a and b and a["diff"] != "" and b["diff"] != "":
            extra = "; ".join(f"{s} {main_rows[(s, f'{v} - {ref}')]['estimate [95% CI]']}"
                              for s in ("full-shear", "full-all") if (s, f"{v} - {ref}") in main_rows)
            md.append(f"- {v} - {ref}, rel_vy: reduced {a['estimate [95% CI]']}, full-reg (main) "
                      f"{b['estimate [95% CI]']} (descriptive: {extra}).")
    md += ["", "Figure: `" + str(args.fig_dir / "bigdata_rel_vy.pdf") + "`.", ""]
    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
