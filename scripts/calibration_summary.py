"""Summary of the training-budget calibration (docs/lab-notebook.md, pre-registered criterion).

Reads the run directories (config.json, metrics.csv, eval_val.json) and the /usr/bin/time logs,
prints per-run and per-budget tables, sigma_ref, Delta1, Delta2, the outcome of the decision rule and
the overfitting check, and writes docs/results/calibration.csv and docs/figures/calibration.png.
Validation only; the test set is not read.

Usage:
    python scripts/calibration_summary.py [--tinfra ~/runs/tinfra] [--calib ~/runs/calib]
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
METRICS = ["rel_all", "rel_p", "rel_vx", "rel_vy", "fluid_rel_all"]
BUDGETS = [17, 67, 133]
COLORS = {17: "#2a78d6", 67: "#eb6834", 133: "#1baf7a"}  # categorical slots 1-3


def run_list(tinfra, calib):
    """(run id, epochs, seed, run dir, time log, used in the decision rule)."""
    runs = [("new_s0_a", 17, 0, True), ("new_s1", 17, 1, True),
            ("new_s0_b", 17, 0, False), ("new_s2", 17, 2, False)]
    out = [(r, e, s, tinfra / r, tinfra / f"{r}.log", used) for r, e, s, used in runs]
    for e in (67, 133):
        for s in (0, 1):
            r = f"ep{e}_s{s}"
            out.append((r, e, s, calib / r, calib / f"{r}.log", True))
    return out


def wall_seconds(log):
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", log.read_text())
    parts = [float(p) for p in m.group(1).split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def max_rss_kb(log):
    return int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", log.read_text()).group(1))


def load_run(rid, epochs, seed, d, log, used):
    cfg = json.loads((d / "config.json").read_text())
    ev = json.loads((d / "eval_val.json").read_text())
    with open(d / "metrics.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert cfg["status"] == "finished" and cfg["epochs"] == epochs and cfg["seed"] == seed, rid
    assert ev["split"] == "val" and ev["n_samples"] == 400, rid
    assert len(rows) == epochs, rid
    return {
        "id": rid, "epochs": epochs, "seed": seed, "used": used,
        "iterations": cfg["effective_iterations"], "commit": cfg["git"]["commit"][:7],
        "dirty": cfg["git"]["dirty"],
        **{m: 100 * ev["means"][m] for m in METRICS},
        "it_s": cfg["it_per_s_train"], "in_script_s": cfg["timing_seconds"]["total_in_script"],
        "wall_s": wall_seconds(log), "rss_gib": max_rss_kb(log) / 2**20,
        "vram_alloc": cfg["peak_vram"]["max_memory_allocated_mib"],
        "vram_res": cfg["peak_vram"]["max_memory_reserved_mib"],
        "train_l1": np.array([float(r["train_L1_mean_of_batches"]) for r in rows]),
        "val_l1": np.array([float(r["val_L1_mean_of_batches"]) for r in rows]),
    }


def overfit(run):
    """Pre-registered overfitting check (reported only, not part of the decision rule)."""
    n = run["epochs"]
    k = max(3, round(0.15 * n))
    v, t = run["val_l1"], run["train_l1"]
    ma = np.convolve(v, np.ones(k) / k, mode="valid")
    last_v = v[-k:].mean()
    cond_a = last_v > 1.05 * ma.min()
    cond_b = t[-k:].mean() < t[-2 * k:-k].mean()
    return {"k": k, "last_over_min": last_v / ma.min(), "final_over_min": v[-1] / ma.min(),
            "argmin_ma": int(np.argmin(ma)) + k - 1, "train_down": cond_b, "flag": cond_a and cond_b}


def fmt_time(s):
    return f"{int(s // 60)}:{s % 60:04.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tinfra", type=Path, default=Path.home() / "runs" / "tinfra")
    ap.add_argument("--calib", type=Path, default=Path.home() / "runs" / "calib")
    args = ap.parse_args()
    runs = [load_run(*r) for r in run_list(args.tinfra, args.calib)]

    print("Per run (val, %; it/s = training loop only; wall = /usr/bin/time; RSS = max RSS)")
    print(f"{'run':9s} {'ep':>3s} {'s':>1s} {'commit':7s} {'rel_all':>7s} {'rel_p':>7s} {'rel_vx':>6s} "
          f"{'rel_vy':>6s} {'it/s':>6s} {'wall':>7s} {'RSS GiB':>7s} {'VRAM':>5s} rule")
    for r in runs:
        print(f"{r['id']:9s} {r['epochs']:3d} {r['seed']:1d} {r['commit']:7s} {r['rel_all']:7.4f} "
              f"{r['rel_p']:7.3f} {r['rel_vx']:6.3f} {r['rel_vy']:6.3f} {r['it_s']:6.1f} "
              f"{fmt_time(r['wall_s']):>7s} {r['rss_gib']:7.2f} {r['vram_res']:5.0f} "
              f"{'yes' if r['used'] else 'no'}")
    dirty = [r["id"] for r in runs if r["dirty"]]
    print(f"Runs with dirty tree: {dirty or 'none'}")

    sigma_ref = np.std([r["rel_all"] for r in runs if r["epochs"] == 17], ddof=1)
    rows = []
    for b in BUDGETS:
        g = [r for r in runs if r["epochs"] == b and r["used"]]
        assert sorted(r["seed"] for r in g) == [0, 1], b
        g.sort(key=lambda r: r["seed"])
        row = {"epochs": b, "effective_iterations": g[0]["iterations"], "seeds": "0;1"}
        for m in METRICS:
            row[f"val_{m}_mean"] = np.mean([r[m] for r in g])
            row[f"val_{m}_absdiff"] = abs(g[0][m] - g[1][m])
        row["it_per_s_train_mean"] = np.mean([r["it_s"] for r in g])
        row["wall_s_mean"] = np.mean([r["wall_s"] for r in g])
        row["in_script_s_mean"] = np.mean([r["in_script_s"] for r in g])
        row["peak_rss_gib_max"] = max(r["rss_gib"] for r in g)
        row["peak_vram_reserved_mib_max"] = max(r["vram_res"] for r in g)
        rows.append(row)

    print("\nPer budget, seeds 0 and 1 (val, %): mean (|s0 - s1|)")
    print(f"{'ep':>3s} {'rel_all':>16s} {'rel_p':>16s} {'rel_vx':>15s} {'rel_vy':>15s} {'fluid_all':>9s}")
    for row in rows:
        cells = [f"{row[f'val_{m}_mean']:.4f} ({row[f'val_{m}_absdiff']:.4f})" for m in METRICS[:4]]
        print(f"{row['epochs']:3d} {cells[0]:>16s} {cells[1]:>16s} {cells[2]:>15s} {cells[3]:>15s} "
              f"{row['val_fluid_rel_all_mean']:9.4f}")
    print(f"\n{'ep':>3s} {'it/s':>6s} {'wall/run':>9s} {'in-script':>9s} {'RSS GiB':>7s} {'VRAM MiB':>8s}")
    for row in rows:
        print(f"{row['epochs']:3d} {row['it_per_s_train_mean']:6.1f} {fmt_time(row['wall_s_mean']):>9s} "
              f"{fmt_time(row['in_script_s_mean']):>9s} {row['peak_rss_gib_max']:7.2f} "
              f"{row['peak_vram_reserved_mib_max']:8.0f}")

    m = {row["epochs"]: row["val_rel_all_mean"] for row in rows}
    d1, d2 = m[17] - m[67], m[67] - m[133]
    if d1 < sigma_ref:
        choice = 17
    elif d2 < sigma_ref:
        choice = 67
    else:
        choice = 133
    print(f"\nsigma_ref (ddof=1, 4 runs at 17 ep.) = {sigma_ref:.4f}")
    print(f"Delta1 = mean(17) - mean(67)  = {d1:.4f}  {'<' if d1 < sigma_ref else '>='} sigma_ref")
    print(f"Delta2 = mean(67) - mean(133) = {d2:.4f}  {'<' if d2 < sigma_ref else '>='} sigma_ref")
    print(f"Rule outcome: {choice} epochs")

    print("\nOverfitting check (pre-registered; '*' = chosen budget)")
    print(f"{'run':9s} {'ep':>3s} {'k':>2s} {'lastk/min':>9s} {'final/min':>9s} {'argmin':>6s} "
          f"{'train down':>10s} flag")
    for r in runs:
        o = overfit(r)
        star = "*" if r["epochs"] == choice else " "
        print(f"{r['id']:9s} {r['epochs']:3d} {o['k']:2d} {o['last_over_min']:9.4f} "
              f"{o['final_over_min']:9.4f} {o['argmin_ma']:6d} {str(o['train_down']):>10s} "
              f"{'POSSIBLE OVERFITTING' if o['flag'] else 'no'}{star}")

    for row in rows:
        row.update({"sigma_ref": sigma_ref, "delta_to_next": {17: d1, 67: d2, 133: float("nan")}[row["epochs"]],
                    "chosen": row["epochs"] == choice})
    out_csv = REPO / "docs" / "results" / "calibration.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"\nWrote {out_csv.relative_to(REPO)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {0: "-", 1: "--", 2: ":"}
    for r in runs:
        ep = np.arange(1, r["epochs"] + 1)
        label = f"{r['epochs']} ep., seed {r['seed']}" + ("" if r["used"] else f" ({r['id']})")
        ax1.plot(ep, r["val_l1"], styles[r["seed"]], color=COLORS[r["epochs"]], lw=1.5 if r["used"] else 1.0,
                 alpha=1.0 if r["used"] else 0.6, label=label)
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("validation L1 (mean of batches, normalised)")
    ax1.set_title("Validation L1 per epoch")
    ax1.legend(fontsize=7, frameon=False)
    ax1.grid(alpha=0.25, lw=0.5)

    for row in rows:
        b = row["epochs"]
        pts = [r["rel_all"] for r in runs if r["epochs"] == b and r["used"]]
        ax2.scatter([b] * len(pts), pts, s=36, color=COLORS[b], edgecolor="white", lw=1, zorder=3)
        ax2.errorbar(b + 4, row["val_rel_all_mean"], yerr=sigma_ref, fmt="D", ms=6, color="#52514e",
                     capsize=3, lw=1, zorder=2)
    ax2.plot([b + 4 for b in BUDGETS], [m[b] for b in BUDGETS], color="#52514e", lw=1, alpha=0.6)
    ax2.set_xticks(BUDGETS)
    ax2.set_xlabel("training budget (epochs)")
    ax2.set_ylabel("final validation rel_all (%)")
    ax2.set_title(f"Seeds 0, 1 (dots); mean +/- sigma_ref (diamonds)\nrule outcome: {choice} epochs",
                  fontsize=10)
    ax2.grid(alpha=0.25, lw=0.5)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_png = REPO / "docs" / "figures" / "calibration.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"Wrote {out_png.relative_to(REPO)}")


if __name__ == "__main__":
    main()
