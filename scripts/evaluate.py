"""Evaluate a checkpoint on the validation split of its run (config.json split_mode) or on the test set.

Primary metrics reproduce train/runTest.py exactly (normalised fields, whole 128x128 grid
including the airfoil interior, ratio of sums per sample, then mean over samples):
  rel_p, rel_vx, rel_vy   sum|o_c - t_c| / sum|t_c| for one channel
  rel_v                   vx and vy pooled: (sum|dvx| + sum|dvy|) / (sum|tvx| + sum|tvy|)  (original "v")
  rel_all                 all three channels pooled (original "combined")
  L1                      nn.L1Loss on normalised fields (original "L1 error")
  L1_denorm               nn.L1Loss after TurbDataset.denormalize (original "Denormalized error";
                          v_norm taken from the normalised inputs, as in the original)
Secondary metrics, prefixed fluid_: the same ratios restricted to fluid cells (input channel 2 == 0;
the channel is 1 inside the airfoil, where all targets are 0).
Further secondary metrics (eval_<split>.json "secondary"; the primary ones are unchanged):
  median_<metric>         median over samples of the per-sample ratio (rel_* and fluid_rel_*; p, vx, vy, all)
  agg_<metric>            aggregate ratio over the whole set: sum over samples of sum|o-t| / sum over samples
                          of sum|t| (float64 sums), whole grid and fluid-only
The per-sample CSV also stores the sums (abs_err_<c>, abs_tgt_<c>, fluid_abs_err_<c>, fluid_abs_tgt_<c>,
float64) that the aggregate ratios are built from.

The test set is only for the pipeline equivalence check and for final results; never use it
to select protocols, hyper-parameters or designs (docs/lab-notebook.md).

Outputs (in the run directory, or --out-dir): eval_<split>_samples.csv (one row per sample, all
metrics) and eval_<split>.json (means over samples, metadata).

Split val_seen (docs/lab-notebook.md, 2026-10-15): seen-shape validation of a run trained with split mode
grouped-holdout (splits/val_seen_holdout_seed0.txt); outputs eval_val_seen.json / eval_val_seen_samples.csv. For such a
run, --split val is its unseen-shape validation (the grouped one).

Usage:
    python scripts/evaluate.py --run-dir DIR --split {val,val_seen,test}
    python scripts/evaluate.py --checkpoint FILE --out-dir DIR --split {val,test} [--model dfpnet --expo 5]

The model is rebuilt with mdf.models.build_from_cfg from the run's config.json (variant, level, channels,
mamba_impl); runs from before the variants (no `variant` field) are read as B.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `mdf`

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mdf import data as mdata
from mdf import models, runinfo

CHANNELS = {"p": [0], "vx": [1], "vy": [2], "v": [1, 2], "all": [0, 1, 2]}
METRICS = ([f"rel_{k}" for k in CHANNELS] + [f"fluid_rel_{k}" for k in CHANNELS] + ["L1", "L1_denorm"])
SECONDARY_CHANNELS = ["p", "vx", "vy", "all"]
SUMS = [f"{pre}abs_{kind}_{k}" for pre in ("", "fluid_") for kind in ("err", "tgt") for k in SECONDARY_CHANNELS]


def abs_sums(out, tgt, chans, mask=None):
    """(sum|o-t|, sum|t|) over the given channels, float64 accumulation."""
    err = tot = 0.0
    for c in chans:
        d = np.abs(out[c].astype(np.float64) - tgt[c].astype(np.float64))
        t = np.abs(tgt[c].astype(np.float64))
        if mask is not None:
            d, t = d[mask], t[mask]
        err += float(d.sum())
        tot += float(t.sum())
    return err, tot


def rel_error(out, tgt, chans, mask=None):
    """Ratio of sums of absolute values over the given channels (float32 numpy, as the original)."""
    if mask is None and len(chans) == 3:
        # original "combined": one sum over the whole (3, 128, 128) array
        return float(np.sum(np.abs(out - tgt)) / np.sum(np.abs(tgt)))
    num = np.float32(0.0)
    den = np.float32(0.0)
    for c in chans:
        d = np.abs(out[c] - tgt[c])
        t = np.abs(tgt[c])
        if mask is not None:
            d, t = d[mask], t[mask]
        num += np.sum(d)
        den += np.sum(t)
    return float(num / den)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["val", "val_seen", "test"], required=True)
    ap.add_argument("--run-dir", type=Path, help="run directory with model.pt (+ config.json)")
    ap.add_argument("--checkpoint", type=Path, help="state_dict file (alternative to --run-dir)")
    ap.add_argument("--out-dir", type=Path, help="output directory (default: the run directory)")
    ap.add_argument("--model", default=None, help="legacy (B only); default: from config.json")
    ap.add_argument("--expo", type=int, default=None, help="legacy (B only); default: from config.json, else 5")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.run_dir is None and args.checkpoint is None:
        sys.exit("Give --run-dir or --checkpoint")
    ckpt = args.checkpoint or args.run_dir / "model.pt"
    out_dir = args.out_dir or args.run_dir or ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {}
    cfg_path = (args.run_dir or ckpt.parent) / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    if args.model or args.expo is not None:
        if cfg.get("variant") not in (None, "B"):
            sys.exit("--model / --expo only apply to variant B")
        expo = args.expo if args.expo is not None else 5
        mcfg = models.normalize_cfg({"variant": "B", "channels": int(2 ** expo + 0.5)})
    else:
        mcfg = models.model_cfg_from_run_config(cfg)
    print(f"Model: {mcfg}")

    if args.split == "test":
        print("NOTE: test set - for the equivalence check and final results only, never for selection.")
        ds = mdata.load_test()
    elif args.split == "val_seen":
        split_mode, fold = cfg.get("split", {}).get("split_mode"), None
        if split_mode != mdata.HOLDOUT:
            sys.exit(f"--split val_seen needs a run trained with split mode {mdata.HOLDOUT} (got {split_mode})")
        ds = mdata.load_val_seen(split_file=mdata.seen_split_file_path())
        print(f"Seen-shape validation of split {split_mode} ({len(ds)} samples)")
        if cfg["split"].get("val_seen_files_sha256") != mdata.file_list_sha256(ds.files):
            sys.exit("Seen-shape validation differs from the one recorded in config.json")
    else:
        # split of the run (config.json); runs from before the grouped split have no split_mode: random
        split_mode = cfg.get("split", {}).get("split_mode", "random") if cfg else "grouped"
        n_folds, fold = cfg.get("split", {}).get("n_folds"), cfg.get("split", {}).get("fold")
        ds = mdata.load_val(split_file=mdata.split_file_path(split_mode, n_folds=n_folds, fold=fold),
                            split_mode=split_mode, n_folds=n_folds, fold=fold)
        print(f"Validation split: {split_mode} ({len(ds)} samples)")
        expected = cfg.get("split", {}).get("val_files_sha256")
        if expected and expected != mdata.file_list_sha256(ds.files):
            sys.exit("Validation split differs from the one recorded in config.json")
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    net = models.build_from_cfg(mcfg)
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    net.cuda().eval()
    criterion = nn.L1Loss().cuda()

    rows = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs_g, targets_g = inputs.cuda(), targets.cuda()
            outputs_g = net(inputs_g)
            l1 = criterion(outputs_g, targets_g).item()
            out = outputs_g.cpu().numpy()[0]
            tgt = targets.numpy()[0]
            inp = inputs.numpy()[0]
            fluid = inp[2] == 0

            row = {"index": i, "file": ds.files[i]}
            for k, chans in CHANNELS.items():
                row[f"rel_{k}"] = rel_error(out, tgt, chans)
                row[f"fluid_rel_{k}"] = rel_error(out, tgt, chans, fluid)
            for k in SECONDARY_CHANNELS:
                for pre, mask in (("", None), ("fluid_", fluid)):
                    row[f"{pre}abs_err_{k}"], row[f"{pre}abs_tgt_{k}"] = abs_sums(out, tgt, CHANNELS[k], mask)
            row["L1"] = l1

            v_norm = mdata.v_norm_from_normalized_input(inp)
            out_dn = torch.from_numpy(mdata.denormalize(out, v_norm)[None]).float().cuda()
            tgt_dn = torch.from_numpy(mdata.denormalize(tgt, v_norm)[None]).float().cuda()
            row["L1_denorm"] = criterion(out_dn, tgt_dn).item()
            row["fluid_cells"] = int(fluid.sum())
            rows.append(row)

    csv_path = out_dir / f"eval_{args.split}_samples.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["index", "file", *METRICS, "fluid_cells", *SUMS])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})

    means = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}
    secondary = {}
    for pre in ("", "fluid_"):
        for k in SECONDARY_CHANNELS:
            secondary[f"median_{pre}rel_{k}"] = float(np.median([r[f"{pre}rel_{k}"] for r in rows]))
            secondary[f"agg_{pre}rel_{k}"] = (sum(r[f"{pre}abs_err_{k}"] for r in rows)
                                             / sum(r[f"{pre}abs_tgt_{k}"] for r in rows))
    summary = {
        "split": args.split,
        "split_mode": split_mode if args.split in ("val", "val_seen") else None,
        "fold": fold if args.split in ("val", "val_seen") else None,
        "n_samples": len(rows),
        "checkpoint": str(ckpt),
        "model": mcfg["variant"],
        "model_cfg": mcfg,
        "trainable_params": models.count_params(net),
        "means": means,
        "secondary": secondary,
        "definitions": {
            "rel_*": "per-sample sum|o-t|/sum|t| on normalised fields, whole grid incl. airfoil interior "
                     "(train/runTest.py), averaged over samples",
            "fluid_rel_*": "same, restricted to cells with input mask channel == 0 (fluid)",
            "rel_v": "vx and vy pooled (original 'v')",
            "rel_all": "p, vx, vy pooled (original 'combined')",
            "median_*": "median over samples of the per-sample ratio (secondary)",
            "agg_*": "sum over all samples of sum|o-t| / sum over all samples of sum|t|, float64 (secondary)",
        },
        "git": runinfo.git_info(),
        "env": runinfo.env_info(),
        "evaluated_utc": runinfo.utc_now(),
    }
    runinfo.write_json(out_dir / f"eval_{args.split}.json", summary)

    print(f"{args.split}: {len(rows)} samples, checkpoint {ckpt}")
    print("Relative error % (original definition, whole grid):")
    print("  p {:.6f}  vx {:.6f}  vy {:.6f}  v {:.6f}  combined {:.6f}".format(
        *(100 * means[f"rel_{k}"] for k in CHANNELS)))
    print("Relative error % (secondary, fluid cells only):")
    print("  p {:.6f}  vx {:.6f}  vy {:.6f}  v {:.6f}  combined {:.6f}".format(
        *(100 * means[f"fluid_rel_{k}"] for k in CHANNELS)))
    print(f"L1 {means['L1']:.6f}  denormalized L1 {means['L1_denorm']:.6f}")
    for pre, label in (("", "whole grid"), ("fluid_", "fluid only")):
        for kind in ("median", "agg"):
            print(f"Secondary {kind} ({label}) %: " + "  ".join(
                f"{k} {100 * secondary[f'{kind}_{pre}rel_{k}']:.6f}" for k in SECONDARY_CHANNELS))
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
