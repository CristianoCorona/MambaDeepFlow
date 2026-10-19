"""Duration and cost estimate of the big-dataset queue from the short cloud check (scripts/cloud/smoke_test.sh).

Inputs: the smoke runs (one per configuration: B, MDv64-local, MDv64; trained with --split-mode grouped-full
--train-fraction F --total-steps N, then evaluated on the full validation) and the full split sizes (inspect_full.py
--json). For each configuration, one 133-epoch run on the full set:
  steps      = 133 * floor(n_train / batch)                      (drop_last, as train.py)
  train      = steps / it_per_s_train of the smoke run           (same GPU, same per-step work)
  validation = 133 * the smoke run's validation pass (the smoke run already validates on the full validation set)
  data load  = smoke data load * (n_train + n_val) / (smoke n_train + n_val)
  evaluation = wall time of the smoke run's evaluate.py (full validation; from smoke_timing.json)
  setup      = 60 s (process start, CUDA context, checkpoint)
The estimate is per run; est= (used by run_queue.py for the cutoff) = estimate x --margin (default 1.15).
Cost = hours x --hourly-usd (read it in the console: see docs/cloud-guide.md; no price is assumed here).

Budget rule (docs/preregistration-bigdata.md): the queue runs in priority order (queues/bigdata.txt); MDv64 runs are
launched only if the estimate of everything before them, plus them, fits in the remaining budget. The script prints
how many runs of the queue fit, and a CUTOFF (absolute local date-time) for launch_bigdata.sh: start + the hours the
remaining budget buys after the reserve (--reserve-usd, default 5: disk, post-processing, sync, idle time).

Writes --est-json (seconds per configuration, used by launch_bigdata.sh to fill est= in the effective queue).

Usage:
    python scripts/cloud/estimate_budget.py --smoke-root ~/runs/bigdata-smoke --inspect-json FILE
        --hourly-usd X [--budget-usd 50] [--spent-usd 0] [--reserve-usd 5] [--est-json FILE]
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/, for run_queue

from run_queue import parse_queue

REPO = Path(__file__).resolve().parent.parent.parent
EPOCHS = 133
SETUP_S = 60.0
# smoke run directory name -> configuration label of the queue (queues/bigdata.txt run names "<label>_s<seed>")
SMOKE_RUNS = {"B": "B", "MDv64-local": "MDv64-local", "MDv64": "MDv64"}


def read_json(p):
    return json.loads(Path(p).read_text())


def smoke_measure(run_dir):
    cfg = read_json(run_dir / "config.json")
    if cfg.get("status") != "finished":
        raise SystemExit(f"{run_dir}: smoke run not finished")
    with open(run_dir / "metrics.csv") as fh:
        rows = list(csv.DictReader(fh))
    timing = read_json(run_dir / "smoke_timing.json") if (run_dir / "smoke_timing.json").exists() else {}
    return {"it_per_s": cfg["it_per_s_train"], "val_pass_s": float(rows[-1]["val_seconds"]),
            "data_load_s": cfg["timing_seconds"]["data_load"], "n_train": cfg["split"]["n_train"],
            "n_val": cfg["split"]["n_val"], "batch": cfg["batch_size"], "eval_s": timing.get("eval_wall_s", 0.0),
            "peak_vram_mib": cfg["peak_vram"]["max_memory_reserved_mib"], "peak_rss_mib": cfg["peak_rss_mib"],
            "gpu": cfg["env"].get("gpu")}


def estimate_run(m, n_train, n_val, epochs=EPOCHS):
    steps = epochs * (n_train // m["batch"])
    parts = {"train": steps / m["it_per_s"], "validation": epochs * m["val_pass_s"],
             "data_load": m["data_load_s"] * (n_train + n_val) / (m["n_train"] + m["n_val"]),
             "evaluation": m["eval_s"], "setup": SETUP_S}
    return steps, parts, sum(parts.values())


def queue_plan(queue_runs, est_by_label, hourly, remaining, reserve):
    """Cumulative hours / cost in queue order; a run fits if the cumulative cost stays within remaining - reserve."""
    plan, cum, fits_all = [], 0.0, True
    for r in queue_runs:
        label = r["name"].rsplit("_s", 1)[0]
        sec = est_by_label[label]
        cum += sec
        cost = cum / 3600 * hourly
        fits = fits_all and cost <= remaining - reserve
        fits_all = fits
        plan.append({"name": r["name"], "hours": sec / 3600, "cum_hours": cum / 3600, "cum_usd": cost, "fits": fits})
    return plan


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke-root", type=Path, required=True)
    ap.add_argument("--inspect-json", type=Path, required=True)
    ap.add_argument("--queue", type=Path, default=REPO / "queues" / "bigdata.txt")
    ap.add_argument("--hourly-usd", type=float, default=None, help="VM price per hour (GPU + machine), from the console")
    ap.add_argument("--budget-usd", type=float, default=50.0)
    ap.add_argument("--spent-usd", type=float, default=0.0, help="credit already used (Billing > Reports)")
    ap.add_argument("--reserve-usd", type=float, default=5.0)
    ap.add_argument("--margin", type=float, default=1.15, help="est= = estimate x margin (worst case for the cutoff)")
    ap.add_argument("--est-json", type=Path, default=None)
    ap.add_argument("--now", default=None, help="start time for the suggested cutoff (default: now)")
    return ap.parse_args()


def main():
    args = parse_args()
    split = read_json(args.inspect_json)["split"]
    n_train, n_val = split["n_train"], split["n_val"]
    queue_runs = parse_queue(args.queue)
    print(f"Full split: train {n_train}, val {n_val}; {EPOCHS} epochs = {EPOCHS * (n_train // 10)} steps at batch 10")
    est, est_worst = {}, {}
    print(f"{'config':12s} {'GPU':>18s} {'it/s':>7s} {'val pass s':>10s} {'VRAM MiB':>9s} {'RSS MiB':>8s} "
          f"{'train h':>8s} {'val h':>6s} {'total h':>8s}")
    for d, label in SMOKE_RUNS.items():
        rd = args.smoke_root / d
        if not (rd / "config.json").exists():
            print(f"{label:12s} smoke run missing ({rd})")
            continue
        m = smoke_measure(rd)
        steps, parts, total = estimate_run(m, n_train, n_val)
        est[label], est_worst[label] = total, total * args.margin
        print(f"{label:12s} {str(m['gpu'])[:18]:>18s} {m['it_per_s']:7.1f} {m['val_pass_s']:10.1f} "
              f"{m['peak_vram_mib']:9.0f} {m['peak_rss_mib']:8.0f} {parts['train'] / 3600:8.2f} "
              f"{parts['validation'] / 3600:6.2f} {total / 3600:8.2f}")
    labels_needed = {r["name"].rsplit("_s", 1)[0] for r in queue_runs}
    if not labels_needed <= set(est):
        sys.exit(f"missing smoke runs for {sorted(labels_needed - set(est))}")
    total_q = sum(est[r["name"].rsplit("_s", 1)[0]] for r in queue_runs)
    print(f"Whole queue ({len(queue_runs)} runs): {total_q / 3600:.1f} h nominal, "
          f"{total_q * args.margin / 3600:.1f} h with margin x{args.margin}")
    if args.est_json:
        args.est_json.write_text(json.dumps({"seconds_nominal": est, "seconds_est": est_worst, "margin": args.margin,
                                             "n_train": n_train, "n_val": n_val}, indent=2) + "\n")
        print(f"Wrote {args.est_json} (est= for launch_bigdata.sh)")
    if args.hourly_usd is None:
        print("No --hourly-usd: cost and cutoff not computed. Read the hourly price of the VM in the console "
              "(docs/cloud-guide.md, 'Prices') and run again with --hourly-usd X.")
        return
    remaining = args.budget_usd - args.spent_usd
    plan = queue_plan(queue_runs, est_worst, args.hourly_usd, remaining, args.reserve_usd)
    print(f"Budget {args.budget_usd:.2f} USD, spent {args.spent_usd:.2f}, remaining {remaining:.2f}, reserve "
          f"{args.reserve_usd:.2f}; {args.hourly_usd:.3f} USD/h (with margin x{args.margin}):")
    for p in plan:
        print(f"  {p['name']:16s} {p['hours']:6.2f} h  cumulative {p['cum_hours']:6.2f} h  {p['cum_usd']:7.2f} USD  "
              f"{'fits' if p['fits'] else 'DOES NOT FIT'}")
    n_fit = sum(p["fits"] for p in plan)
    mdv64_fit = all(p["fits"] for p in plan if p["name"].startswith("MDv64_"))
    print(f"Runs that fit: {n_fit} of {len(plan)}; MDv64 runs {'all fit' if mdv64_fit else 'do NOT all fit'}")
    hours = max(0.0, (remaining - args.reserve_usd) / args.hourly_usd)
    now = datetime.fromisoformat(args.now) if args.now else datetime.now()
    cut = now + timedelta(hours=hours)
    last = plan[n_fit - 1]["cum_hours"] if n_fit else 0.0
    print(f"Hours the budget buys after the reserve: {hours:.1f} h. Suggested cutoff (no run starts that would end "
          f"later): CUTOFF={cut.strftime('%Y-%m-%dT%H:%M')}")
    print(f"Expected end of the runs that fit (with margin): {(now + timedelta(hours=last)).strftime('%Y-%m-%d %H:%M')}"
          f" (+ post-processing, then automatic shutdown)")
    if n_fit < len(plan):
        print("Runs that do not fit are not launched and are declared in the notebook (pre-registration).")
    if math.isfinite(hours) and n_fit == 0:
        print("WARNING: not even the first run fits the budget.")


if __name__ == "__main__":
    main()
