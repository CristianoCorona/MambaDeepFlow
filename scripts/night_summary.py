"""Summary of night 1: protocol confirmation on the profile-disjoint validation split.

Reads config.json, eval_val.json and queue_timing.json of the night-1 runs, prints per-run values, per-budget
mean and seed difference, applies the pre-registered criterion (docs/lab-notebook.md, "Pre-registered criterion:
protocol confirmation on the profile-disjoint split") and compares the error level with the calibration runs on
the random validation split. Writes docs/results/night1.csv. Validation only; the test set is not read.

Usage:
    python scripts/night_summary.py [--night ~/runs/night1] [--calib ~/runs/calib]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
METRICS = ["rel_all", "rel_p", "rel_vx", "rel_vy", "fluid_rel_all"]
GROUPED_SHA = "3b26466ac7ccc1ad6db6f9a0f2cbaffee01d11784e7d537d5e139e5af085e2e6"
RANDOM_SHA = "7f6bc466c7937c2bf6e52c3e47afffa07e21c0c2e72cfe26fb05de9a75342023"


def load_run(run_dir, expect_sha):
    cfg = json.loads((run_dir / "config.json").read_text())
    ev = json.loads((run_dir / "eval_val.json").read_text())
    timing_file = run_dir / "queue_timing.json"
    timing = json.loads(timing_file.read_text()) if timing_file.exists() else {}
    assert cfg["status"] == "finished", run_dir
    assert cfg["split"]["val_files_sha256"] == expect_sha, run_dir
    assert not cfg["git"]["dirty"], run_dir
    return {
        "run": run_dir.name, "epochs": cfg["epochs"], "seed": cfg["seed"],
        "split": cfg["split"].get("split_mode", "random"), "n_val": ev["n_samples"],
        "effective_iterations": cfg["effective_iterations"], "commit": cfg["git"]["commit"][:7],
        **{m: 100 * ev["means"][m] for m in METRICS},
        "it_per_s": cfg["it_per_s_train"],
        "train_wall_s": timing.get("train_wall_s", cfg["timing_seconds"]["total_in_script"]),
        "eval_wall_s": timing.get("eval_wall_s"),
    }


def by_budget(runs, epochs, seeds):
    sel = [r for r in runs if r["epochs"] == epochs and r["seed"] in seeds]
    assert sorted(r["seed"] for r in sel) == sorted(seeds), (epochs, seeds)
    return sorted(sel, key=lambda r: r["seed"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--night", type=Path, default=Path.home() / "runs" / "night1")
    ap.add_argument("--calib", type=Path, default=Path.home() / "runs" / "calib")
    args = ap.parse_args()

    night = [load_run(d, GROUPED_SHA) for d in sorted(args.night.iterdir())
             if d.is_dir() and (d / "eval_val.json").exists() and ".interrupted-" not in d.name]
    calib = [load_run(args.calib / f"ep{e}_s{s}", RANDOM_SHA) for e in (67, 133) for s in (0, 1)]

    print("Per run, profile-disjoint validation (402 samples), values in %:")
    for r in night:
        print(f"  {r['run']:9s} ep {r['epochs']:3d} seed {r['seed']}  rel_all {r['rel_all']:.4f}  "
              f"rel_p {r['rel_p']:.4f}  rel_vx {r['rel_vx']:.4f}  rel_vy {r['rel_vy']:.4f}  "
              f"fluid_rel_all {r['fluid_rel_all']:.4f}")
        print(f"  {'':9s} train {r['train_wall_s'] / 60:.2f} min, eval {r['eval_wall_s']:.0f} s, "
              f"{r['it_per_s']:.1f} it/s, {r['effective_iterations']} it, commit {r['commit']}")

    print("\nPer budget, seeds 0 and 1: mean (|s0 - s1|), %:")
    means = {}
    for e in (67, 133):
        sel = by_budget(night, e, (0, 1))
        means[e] = {m: np.mean([r[m] for r in sel]) for m in METRICS}
        print(f"  {e:3d} epochs: " + "  ".join(
            f"{m} {means[e][m]:.4f} ({abs(sel[0][m] - sel[1][m]):.4f})" for m in METRICS))
    s3 = by_budget(night, 133, (0, 1, 2))
    print("  133 epochs, seeds 0-2: " + "  ".join(
        f"{m} {np.mean([r[m] for r in s3]):.4f} (range {np.ptp([r[m] for r in s3]):.4f}, "
        f"sd {np.std([r[m] for r in s3], ddof=1):.4f})" for m in METRICS[:4]))

    a, b = means[133]["rel_all"], means[67]["rel_all"]
    print(f"\nCriterion: mean val rel_all 133 ep (s0, s1) = {a:.4f} vs 67 ep (s0, s1) = {b:.4f}; "
          f"difference {a - b:+.4f}")
    print("  Outcome: " + ("133 < 67 -> protocol stays at 133 epochs" if a < b
                           else "133 is NOT lower than 67 -> report to the team, decide together"))

    print("\nError level, random validation (calibration, 400 samples) vs profile-disjoint (night 1), "
          "mean of seeds 0 and 1, %:")
    for e in (67, 133):
        cm = {m: np.mean([r[m] for r in by_budget(calib, e, (0, 1))]) for m in METRICS}
        for m in METRICS[:4]:
            print(f"  {e:3d} ep {m:7s}: random {cm[m]:.4f}  grouped {means[e][m]:.4f}  "
                  f"ratio {means[e][m] / cm[m]:.3f}")
    c67 = np.mean([r["rel_all"] for r in by_budget(calib, 67, (0, 1))])
    c133 = np.mean([r["rel_all"] for r in by_budget(calib, 133, (0, 1))])
    print(f"  gain 67 -> 133 ep in rel_all: random {c67 - c133:.4f} ({(c67 - c133) / c67:.1%}), "
          f"grouped {b - a:.4f} ({(b - a) / b:.1%})")

    out = REPO / "docs" / "results" / "night1.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(night[0]))
        w.writeheader()
        for r in night + calib:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nWrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
