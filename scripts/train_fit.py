"""Training error in eval mode vs validation error (verification 5a, docs/lab-notebook.md 2026-10-14).

For every run: final model (model.pt) in eval() mode, inference on the training part of the grouped split (5998
files of data/train; explore.load_split("train") asserts that no file is read from data/test and that train and
validation do not overlap), batch --batch-size (default 10), per-sample sums on the GPU in float64:
  L1_s     mean |o - t| over the 3 x 128 x 128 values (nn.L1Loss of one sample, normalised fields)
  rel_<c>  sum|o_c - t_c| / sum|t_c| (as evaluate.py), c in p, vx, vy, all
then means over samples. The same quantities on validation (402 samples) come from explore.cached_inference
(batch 1, as evaluate.py): L1_s = abs_err_all / (3 * 128 * 128).

Batch 10 vs batch 1 only changes the cuDNN TF32 rounding (notebook 2026-10-13: <= 0.001 points on run means),
negligible against the seed spread.

Pre-registered condition (notebook 2026-10-14, commit 26b6e7f): for MD8 or SA8, both
  (i)  train L1: (mean_var - mean_B) / SE_Welch <= -2,   (ii) val L1: (mean_var - mean_B) / SE_Welch >= -1,
SE_Welch = sqrt(s_var^2/n_var + s_B^2/n_B), std ddof=1 over seeds. Printed and written to --out-dir.

Usage:
    python scripts/train_fit.py --config B=GLOB [--config ...] [--seeds 0-7] --out-dir DIR
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import numpy as np
import torch

import analyze_val as av
from mdf import explore, models, runinfo

NPIX = 3 * 128 * 128
CH = ("p", "vx", "vy")
CONDITION_VARIANTS = ("MD8", "SA8")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / "runs" / "explore-cache")
    return ap.parse_args()


def train_metrics(run_dir, inputs, targets, batch_size):
    cfg = json.loads((run_dir / "config.json").read_text())
    net = models.build_from_cfg(models.model_cfg_from_run_config(cfg))
    net.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    net.cuda().eval()
    n = inputs.shape[0]
    err = torch.zeros(n, 3, dtype=torch.float64)
    tot = torch.zeros(n, 3, dtype=torch.float64)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            x = inputs[s:s + batch_size].cuda()
            t = targets[s:s + batch_size].cuda().double()
            o = net(x).double()
            err[s:s + batch_size] = (o - t).abs().sum(dim=(2, 3)).cpu()
            tot[s:s + batch_size] = t.abs().sum(dim=(2, 3)).cpu()
    err, tot = err.numpy(), tot.numpy()
    out = {"L1": float(np.mean(err.sum(1) / NPIX)), "rel_all": float(np.mean(err.sum(1) / tot.sum(1)))}
    for i, c in enumerate(CH):
        out[f"rel_{c}"] = float(np.mean(err[:, i] / tot[:, i]))
    return out


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    d = a.mean() - b.mean()
    return d, se, d / se


def main():
    args = parse_args()
    runs = av.collect_runs(args.config, av.parse_seeds(args.seeds))
    t0 = time.perf_counter()
    train_ds = explore.load_split("train", "grouped")
    val_ds = explore.load_split("val", "grouped")
    print(f"loaded train {len(train_ds)} / val {len(val_ds)} samples in {time.perf_counter() - t0:.1f} s")
    props = explore.sample_properties(val_ds)
    regions = explore.all_region_masks(val_ds, props)

    rows = []
    gpu_s = 0.0
    for label, lst in runs.items():
        for seed, d, cfg in lst:
            t1 = time.perf_counter()
            tr = train_metrics(d, train_ds.inputs, train_ds.targets, args.batch_size)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t1
            gpu_s += dt
            r, _ = explore.cached_inference(d, val_ds, regions, [], args.cache_dir, "val")
            va = {"L1": float(np.mean(r["abs_err_all"] / NPIX)), "rel_all": float(r["rel_all"].mean()),
                  **{f"rel_{c}": float(r[f"rel_{c}"].mean()) for c in CH}}
            rows.append({"config": label, "seed": seed, **{f"train_{k}": v for k, v in tr.items()},
                         **{f"val_{k}": v for k, v in va.items()}, "gap_L1": va["L1"] - tr["L1"],
                         "train_seconds": dt})
            print(f"{label} s{seed}: train L1 {tr['L1']:.6e} val L1 {va['L1']:.6e} ({dt:.1f} s)", flush=True)
    av.write_csv(args.out_dir / "train_fit_runs.csv", rows)

    labels = list(runs)
    by = {l: [r for r in rows if r["config"] == l] for l in labels}
    summ = []
    for l in labels:
        rr = by[l]
        row = {"config": l, "seeds": len(rr)}
        for k in ("train_L1", "val_L1", "gap_L1"):
            x = 1e3 * np.array([r[k] for r in rr])
            row[f"{k} x1e-3"] = f"{x.mean():.4f} ({x.std(ddof=1):.4f})" if x.size > 1 else f"{x.mean():.4f}"
        for k in ("train_rel_all", "train_rel_p", "train_rel_vx", "train_rel_vy", "val_rel_all"):
            x = 100 * np.array([r[k] for r in rr])
            row[f"{k} %"] = f"{x.mean():.4f} ({x.std(ddof=1):.4f})" if x.size > 1 else f"{x.mean():.4f}"
        if l != "B" and "B" in by:
            for k in ("train_L1", "val_L1"):
                d, se, z = welch([r[k] for r in rr], [r[k] for r in by["B"]])
                row[f"{k} - B x1e-3 (diff/SE)"] = f"{1e3 * d:+.4f} ({z:+.2f})"
        summ.append(row)
    fields = list(dict.fromkeys(k for r in summ for k in r))
    for r in summ:
        for k in fields:
            r.setdefault(k, "")
    av.write_csv(args.out_dir / "train_fit_summary.csv", summ, fields)

    cond_rows, any_ok = [], False
    for v in CONDITION_VARIANTS:
        if v not in by or "B" not in by:
            continue
        _, _, zt = welch([r["train_L1"] for r in by[v]], [r["train_L1"] for r in by["B"]])
        _, _, zv = welch([r["val_L1"] for r in by[v]], [r["val_L1"] for r in by["B"]])
        ok_i, ok_ii = zt <= -2, zv >= -1
        any_ok |= ok_i and ok_ii
        cond_rows.append({"variant": v, "train diff/SE": f"{zt:+.2f}", "(i) <= -2": "yes" if ok_i else "no",
                          "val diff/SE": f"{zv:+.2f}", "(ii) >= -1": "yes" if ok_ii else "no",
                          "both": "yes" if ok_i and ok_ii else "no"})
    md = ["# Training error in eval mode vs validation (verification 5a)", "",
          f"Generated by `scripts/train_fit.py`, git {runinfo.git_info()}, {runinfo.utc_now()}. Train: "
          f"{len(train_ds)} samples, batch {args.batch_size}, eval mode; val: {len(val_ds)} samples, batch 1 "
          f"(cache). GPU inference time on the training set: {gpu_s:.0f} s in total.", "",
          "L1 = mean |o - t| per sample (normalised fields, whole grid), mean over samples; mean (std ddof=1) over "
          "seeds; diff/SE = (mean - mean_B) / sqrt(s^2/n + s_B^2/n_B).", "",
          av.md_table(summ, fields), "",
          "## Pre-registered condition (commit 26b6e7f)", "",
          av.md_table(cond_rows, list(cond_rows[0].keys())) if cond_rows else "(not applicable)", "",
          f"**Condition satisfied: {'YES' if any_ok else 'NO'}** (MD8 or SA8 with both (i) and (ii)).", ""]
    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
