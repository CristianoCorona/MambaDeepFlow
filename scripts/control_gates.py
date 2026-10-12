"""Pass / fail of the control runs (docs/lab-notebook.md, 2026-10-12, pre-registered criterion).

For each run directory under --root (name = configuration):
  (a) no NaN / Inf in the per-epoch train and val L1 of metrics.csv (a NaN / Inf batch loss makes its
      epoch mean NaN / Inf, so the epoch means are sufficient);
  (b) training loss decreasing: mean train L1 of the last 3 epochs < mean of the first 3;
  (c) final val L1 <= 1.15 x final val L1 of the reference run (B).
Also reports it/s (training loop) from config.json, the official training-cost measure.

Usage: python scripts/control_gates.py --root ~/runs/control-2026-10-12 [--ref B] [--csv FILE]
"""

import argparse
import csv
import json
import math
from pathlib import Path

ORDER = ["B", "MD8", "SA8", "CBAM8", "B-large", "MD32", "SA32"]


def load(run_dir):
    cfg = json.loads((run_dir / "config.json").read_text())
    rows = list(csv.DictReader(open(run_dir / "metrics.csv")))
    train = [float(r["train_L1_mean_of_batches"]) for r in rows]
    val = [float(r["val_L1_mean_of_batches"]) for r in rows]
    return cfg, train, val


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--ref", default="B")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    root = args.root.expanduser()

    names = [n for n in ORDER if (root / n / "config.json").exists()]
    names += sorted(p.parent.name for p in root.glob("*/config.json") if p.parent.name not in names
                    and ".interrupted-" not in p.parent.name)
    _, _, ref_val = load(root / args.ref)
    ref_final = ref_val[-1]

    out = []
    print(f"{'run':8} {'status':9} {'ep':>3} {'params':>9} {'it/s':>6} {'train first3':>12} {'train last3':>11} "
          f"{'val L1 final':>12} {'/B':>6}  a b c  verdict")
    for n in names:
        cfg, train, val = load(root / n)
        finite = all(math.isfinite(v) for v in train + val)
        first3, last3 = sum(train[:3]) / 3, sum(train[-3:]) / 3
        a = finite
        b = finite and last3 < first3
        ratio = val[-1] / ref_final
        c = finite and ratio <= 1.15
        ok = cfg.get("status") == "finished" and len(train) == cfg.get("epochs") and a and b and c
        its = cfg.get("it_per_s_train")
        row = {"run": n, "status": cfg.get("status"), "epochs": len(train), "params": cfg.get("trainable_params"),
               "it_per_s_train": its, "train_first3": first3, "train_last3": last3, "val_L1_final": val[-1],
               "ratio_vs_B": ratio, "a": a, "b": b, "c": c, "pass": ok}
        out.append(row)
        print(f"{n:8} {cfg.get('status'):9} {len(train):3d} {cfg.get('trainable_params'):9d} "
              f"{its if its is None else round(its, 1)!s:>6} {first3:12.6f} {last3:11.6f} {val[-1]:12.6f} "
              f"{ratio:6.3f}  {'Y' if a else 'N'} {'Y' if b else 'N'} {'Y' if c else 'N'}  {'PASS' if ok else 'FAIL'}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
