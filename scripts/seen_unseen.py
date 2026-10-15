"""Seen-shape vs unseen-shape validation (EXPLORATORY) and validation calibration on the test set
(docs/lab-notebook.md, 2026-10-15, pre-registration commit d46b279).

Runs: trained with --split-mode grouped-holdout (night 6: B, MD8, MDv8-16-32, seeds 0-2). Two validation sets of the
same split: unseen shapes (= the grouped validation, 402 samples / 85 profiles) and seen shapes (400 samples, one per
profile, every profile also in training).

Part 1 (validation only)
  per configuration and set: mean (sd over seeds) of rel_all, rel_vy, rel_p (per-sample ratio, mean over samples);
  B: ratio unseen / seen (to compare with the x1.31 grouped / random ratio of 2026-10-11);
  difference of differences DoD = (V - B)_seen - (V - B)_unseen for V in MD8, MDv8-16-32: rel_all (hypothesis:
  DoD < 0), rel_vy and rel_p (descriptive). 95 % percentile interval, bootstrap (--n-boot 10 000, seed 0) that resamples
  the seeds of each configuration once per repetition, jointly for both sets, and the profiles separately in each set.
Night 6b (docs/preregistration-night6b.md): also DRv11 and TR (grouped-holdout, seeds 0-4), DoD vs B for each. B
has seeds 0-4 there: MD8 and MDv8-16-32 are compared with B seeds 0-2 only (pre-registered, d46b279; amendment of
night 6b), DRv11 and TR with all available B seeds (REF_SEEDS). The unseen / seen ratio of B uses all its seeds.
Part 2 (only with --split test --allow-test; descriptive, no conclusion on the architectures)
  per configuration: rel_all and rel_vy on seen, unseen and test (means over seeds);
  (a) which validation has the error level closer to the test (|mean_val - mean_test|);
  (b) which validation orders the configurations as the test (order of the means, Kendall tau).

Per-sample values from explore.cached_inference (batch 1, as evaluate.py), checked against eval_<split>_samples.csv
when present. The test set is refused unless --split test --allow-test is given.

Usage:
    python scripts/seen_unseen.py --config B=~/runs/night6/GH_B_s* --config MD8=~/runs/night6/GH_MD8_s* \
        --config MDv8-16-32=~/runs/night6/GH_MDv8-16-32_s* --out-dir docs/results/seen-unseen [--split test --allow-test]
"""

import argparse
import csv
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `mdf`
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/, for analyze_val helpers

import numpy as np
import torch

import analyze_val as av
from mdf import data as mdata
from mdf import explore, runinfo

REF = "B"
VARIANTS = ("MD8", "MDv8-16-32", "DRv11", "TR")
# B seeds each variant is compared with (None = all loaded): MD8 / MDv8-16-32 as pre-registered in d46b279 (B seeds 0-2;
# B seeds 3-4 exist from night 6b, where they serve the DRv11 confirmation and are only descriptive here)
REF_SEEDS = {"MD8": (0, 1, 2), "MDv8-16-32": (0, 1, 2), "DRv11": None, "TR": None}
ORDER = (REF, *VARIANTS)
METRICS = ("rel_all", "rel_vy", "rel_p")  # rel_all: hypothesis; the others descriptive
CALIB_METRICS = ("rel_all", "rel_vy")
SETS = ("seen", "unseen")
SPLIT_OF = {"seen": "val_seen", "unseen": "val", "test": "test"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", action="append", required=True, help="LABEL=GLOB of run directories (repeatable)")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="val: part 1 only; test: also the calibration (part 2), needs --allow-test")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / "runs" / "explore-cache")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        ap.error("--split test is for the single pre-registered test session; pass --allow-test explicitly")
    return args


def seed_weights(rng, k, n_boot):
    pick = rng.integers(0, k, size=(n_boot, k))
    w = np.zeros((n_boot, k))
    np.add.at(w, (np.arange(n_boot)[:, None], pick), 1.0)
    return w / k


def profile_means(d_boot, profiles, rng, n_boot):
    """Sample-weighted mean of d over resampled profiles; d_boot (n_boot, N) or (N,) for the point estimate."""
    g, _ = explore.profile_onehot(profiles)
    counts = g.sum(axis=0)
    idx = rng.integers(0, g.shape[1], size=(n_boot, g.shape[1]))
    prof = d_boot @ g  # (n_boot, P)
    return np.take_along_axis(prof, idx, axis=1).sum(axis=1) / counts[idx].sum(axis=1)


def dod_bootstrap(v, b, profiles, n_boot, seed):
    """v, b: {set: (seeds, N_set) per-sample metric}, rows = the same runs in both sets; profiles: {set: list}.
    Returns point estimates and 95 % intervals of D_seen, D_unseen and DoD = D_seen - D_unseen (V - B)."""
    rng = np.random.default_rng(seed)
    wv, wb = seed_weights(rng, v["seen"].shape[0], n_boot), seed_weights(rng, b["seen"].shape[0], n_boot)
    out, boot = {}, {}
    for s in SETS:
        out[f"D_{s}"] = float(v[s].mean(0).mean() - b[s].mean(0).mean())
        boot[s] = profile_means(wv @ v[s] - wb @ b[s], profiles[s], rng, n_boot)
        out[f"D_{s}_lo"], out[f"D_{s}_hi"] = np.percentile(boot[s], [2.5, 97.5])
    out["DoD"] = out["D_seen"] - out["D_unseen"]
    out["DoD_lo"], out["DoD_hi"] = np.percentile(boot["seen"] - boot["unseen"], [2.5, 97.5])
    return out


def ref_rows(variant, ref_seeds):
    """Row indices of the reference runs (seeds `ref_seeds`, in load order) used in the DoD of `variant`."""
    keep = REF_SEEDS[variant]
    rows = [i for i, s in enumerate(ref_seeds) if keep is None or s in keep]
    if not rows:
        sys.exit(f"{variant}: none of the reference seeds {keep} loaded")
    return rows


def kendall_tau(a, b):
    """Kendall tau between two score lists (no ties expected)."""
    s = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (len(a) * (len(a) - 1) / 2)


def load_values(runs, labels, set_name, ds, args):
    props = explore.sample_properties(ds)
    regions = explore.all_region_masks(ds, props)
    split = SPLIT_OF[set_name]
    vals, max_diff = {}, 0.0
    for label in labels:
        per_run = []
        for seed, d, cfg in runs[label]:
            r, cached = explore.cached_inference(d, ds, regions, [], args.cache_dir, split, args.device)
            ev = d / f"eval_{split}_samples.csv"
            if ev.exists():
                with open(ev) as fh:
                    rows = list(csv.DictReader(fh))
                assert [row["file"] for row in rows] == ds.files, f"{ev}: file order differs"
                max_diff = max(max_diff, max(float(np.max(np.abs(np.array([float(row[m]) for row in rows]) - r[m])))
                                             for m in METRICS))
            per_run.append(r)
            print(f"{set_name} {label} s{seed}: {'cache' if cached else 'inference'}")
        vals[label] = {m: np.stack([r[m] for r in per_run]) for m in METRICS}
    return vals, [p["profile"] for p in props], max_diff


def main():
    args = parse_args()
    runs = av.collect_runs(args.config, av.parse_seeds(args.seeds))
    labels = [l for l in ORDER if l in runs]
    if REF not in runs or not set(VARIANTS) & set(runs):
        sys.exit(f"need {REF} and at least one of {VARIANTS}; got {sorted(runs)}")
    if set(runs) - set(ORDER):
        sys.exit(f"unexpected labels {sorted(set(runs) - set(ORDER))}; expected {ORDER}")
    sets = {"unseen": explore.load_split("val", mdata.HOLDOUT), "seen": explore.load_split("val_seen", mdata.HOLDOUT)}
    for lst in runs.values():
        for seed, d, cfg in lst:
            sp = cfg.get("split", {})
            if sp.get("split_mode") != mdata.HOLDOUT:
                sys.exit(f"{d}: split mode {sp.get('split_mode')}, expected {mdata.HOLDOUT}")
            if (sp["val_files_sha256"], sp["val_seen_files_sha256"]) != (
                    mdata.file_list_sha256(sets["unseen"].files), mdata.file_list_sha256(sets["seen"].files)):
                sys.exit(f"{d}: validation lists differ from the ones recorded in config.json")
    if args.split == "test":
        sets["test"] = explore.load_split("test", allow_test=args.allow_test)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vals, profiles, max_diff = {}, {}, 0.0
    for s, ds in sets.items():
        vals[s], profiles[s], md_ = load_values(runs, labels, s, ds, args)
        max_diff = max(max_diff, md_)
    seeds = {l: [s for s, _, _ in runs[l]] for l in labels}

    # --- T1: means per configuration and set ---
    t1 = []
    for l in labels:
        for s in sets:
            row = {"config": l, "set": s, "seeds": len(seeds[l]), "samples": len(sets[s]),
                   "profiles": len(set(profiles[s]))}
            for m in METRICS:
                x = 100 * vals[s][l][m].mean(axis=1)
                row[f"{m} %"] = f"{x.mean():.4f} ({x.std(ddof=1):.4f})" if x.size > 1 else f"{x.mean():.4f}"
            t1.append(row)
    av.write_csv(args.out_dir / "t1_sets.csv", t1)
    ratio = []
    for m in METRICS:
        per_seed = vals["unseen"][REF][m].mean(1) / vals["seen"][REF][m].mean(1)
        ratio.append({"metric": m, "B unseen / seen (ratio of seed means)":
                      f"{vals['unseen'][REF][m].mean() / vals['seen'][REF][m].mean():.3f}",
                      "per seed": ", ".join(f"{x:.3f}" for x in per_seed)})

    # --- T2: difference of differences ---
    t2 = []
    for v in VARIANTS:
        if v not in runs:
            continue
        rows = ref_rows(v, seeds[REF])
        for m in METRICS:
            r = dod_bootstrap({s: vals[s][v][m] for s in SETS}, {s: vals[s][REF][m][rows] for s in SETS},
                              profiles, args.n_boot, args.boot_seed)
            t2.append({"comparison": f"{v} - {REF}", "metric": m, "role": "hypothesis" if m == "rel_all" else
                       "descriptive", "seeds": f"{len(seeds[v])}/{len(rows)}",
                       **{k: f"{100 * r[k]:+.4f}" for k in ("D_seen", "D_unseen", "DoD")},
                       "D_seen 95% CI": f"[{100 * r['D_seen_lo']:+.4f}, {100 * r['D_seen_hi']:+.4f}]",
                       "D_unseen 95% CI": f"[{100 * r['D_unseen_lo']:+.4f}, {100 * r['D_unseen_hi']:+.4f}]",
                       "DoD 95% CI": f"[{100 * r['DoD_lo']:+.4f}, {100 * r['DoD_hi']:+.4f}]",
                       "DoD upper < 0": "yes" if r["DoD_hi"] < 0 else "no"})
    av.write_csv(args.out_dir / "t2_dod.csv", t2)

    md = ["# Seen-shape vs unseen-shape validation (EXPLORATORY; pre-registration commit d46b279)", "",
          f"Generated by `scripts/seen_unseen.py`, git {runinfo.git_info()}, {runinfo.utc_now()}. Seeds: "
          + "; ".join(f"{l} {seeds[l]}" for l in labels) + f". Bootstrap {args.n_boot} repetitions, seed "
          f"{args.boot_seed}. Values in %.", "",
          f"Consistency: max |recomputed - eval CSV| over all runs, sets and {list(METRICS)}: **{max_diff:.3g}**.", "",
          "## T1 Mean (sd over seeds) per configuration and set", "",
          "unseen = grouped validation (profiles not in training); seen = one file per profile, profile in training.",
          "", av.md_table(t1, list(t1[0].keys())), "",
          "B: unseen / seen ratio (x1.31 for rel_all grouped / random on 2026-10-11, different samples):", "",
          av.md_table(ratio, list(ratio[0].keys())), "",
          "## T2 Difference of differences DoD = (V - B)_seen - (V - B)_unseen (percentage points)", "",
          "Hypothesis (rel_all): DoD < 0, i.e. the variant looks better relative to B on seen shapes. Interval: "
          "seeds of each configuration resampled once per repetition, jointly for both sets; profiles resampled "
          "separately in each set. Exploratory; no correction for two comparisons. Seeds column: variant / B; "
          "MD8 and MDv8-16-32 use B seeds 0-2 (pre-registered), DRv11 and TR all loaded B seeds.", "",
          av.md_table(t2, list(t2[0].keys())) if t2 else "(no variant loaded)", ""]

    # --- Part 2: calibration on the test set ---
    if args.split == "test":
        cal, order_rows = [], []
        for m in CALIB_METRICS:
            means = {s: {l: 100 * vals[s][l][m].mean() for l in labels} for s in ("seen", "unseen", "test")}
            for l in labels:
                gaps = {s: abs(means[s][l] - means["test"][l]) for s in SETS}
                cal.append({"metric": m, "config": l, **{f"{s} %": f"{means[s][l]:.4f}" for s in ("seen", "unseen",
                                                                                                   "test")},
                            "|seen - test|": f"{gaps['seen']:.4f}", "|unseen - test|": f"{gaps['unseen']:.4f}",
                            "(a) closer to test": min(gaps, key=gaps.get),
                            "test / seen": f"{means['test'][l] / means['seen'][l]:.3f}",
                            "test / unseen": f"{means['test'][l] / means['unseen'][l]:.3f}"})
            order = {s: sorted(labels, key=lambda l: means[s][l]) for s in ("seen", "unseen", "test")}
            for s in SETS:
                order_rows.append({"metric": m, "validation": s, "order (best first)": " < ".join(order[s]),
                                   "test order": " < ".join(order["test"]),
                                   "(b) same order as test": "yes" if order[s] == order["test"] else "no",
                                   "Kendall tau vs test": f"{kendall_tau([means[s][l] for l in labels], [means['test'][l] for l in labels]):+.2f}"})
        av.write_csv(args.out_dir / "t3_calibration_level.csv", cal)
        av.write_csv(args.out_dir / "t3_calibration_order.csv", order_rows)
        md += ["## T3 Validation calibration on the test set (descriptive; no conclusion on the architectures)", "",
               f"Means over seeds; test: {len(sets['test'])} samples, {len(set(profiles['test']))} profiles. These "
               "models have 400 fewer training files than the main runs.", "",
               "(a) error level:", "", av.md_table(cal, list(cal[0].keys())), "",
               "(b) order of the configurations:", "", av.md_table(order_rows, list(order_rows[0].keys())), ""]

    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
