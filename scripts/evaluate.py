"""Evaluate a checkpoint on the fixed validation split or on the test set.

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

The test set is only for the pipeline equivalence check and for final results; never use it
to select protocols, hyper-parameters or designs (docs/lab-notebook.md).

Outputs (in the run directory, or --out-dir): eval_<split>_samples.csv (one row per sample, all
metrics) and eval_<split>.json (means over samples, metadata).

Usage:
    python scripts/evaluate.py --run-dir DIR --split {val,test}
    python scripts/evaluate.py --checkpoint FILE --out-dir DIR --split {val,test} [--model dfpnet --expo 5]
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
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--run-dir", type=Path, help="run directory with model.pt (+ config.json)")
    ap.add_argument("--checkpoint", type=Path, help="state_dict file (alternative to --run-dir)")
    ap.add_argument("--out-dir", type=Path, help="output directory (default: the run directory)")
    ap.add_argument("--model", default=None, help="default: from config.json, else dfpnet")
    ap.add_argument("--expo", type=int, default=None, help="default: from config.json, else 5")
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
    model_name = args.model or cfg.get("model", "dfpnet")
    expo = args.expo if args.expo is not None else cfg.get("args", {}).get("expo", 5)

    if args.split == "test":
        print("NOTE: test set - for the equivalence check and final results only, never for selection.")
        ds = mdata.load_test()
    else:
        files = mdata.read_split_file(mdata.SPLITS_DIR / f"val_seed{mdata.SPLIT_SEED}.txt")
        ds = mdata.FlowDataset(mdata.TRAIN_DIR, files)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    net = models.build_model(model_name, expo=expo)
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
            row["L1"] = l1

            v_norm = mdata.v_norm_from_normalized_input(inp)
            out_dn = torch.from_numpy(mdata.denormalize(out, v_norm)[None]).float().cuda()
            tgt_dn = torch.from_numpy(mdata.denormalize(tgt, v_norm)[None]).float().cuda()
            row["L1_denorm"] = criterion(out_dn, tgt_dn).item()
            row["fluid_cells"] = int(fluid.sum())
            rows.append(row)

    csv_path = out_dir / f"eval_{args.split}_samples.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["index", "file", *METRICS, "fluid_cells"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})

    means = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}
    summary = {
        "split": args.split,
        "n_samples": len(rows),
        "checkpoint": str(ckpt),
        "model": model_name,
        "expo": expo,
        "means": means,
        "definitions": {
            "rel_*": "per-sample sum|o-t|/sum|t| on normalised fields, whole grid incl. airfoil interior "
                     "(train/runTest.py), averaged over samples",
            "fluid_rel_*": "same, restricted to cells with input mask channel == 0 (fluid)",
            "rel_v": "vx and vy pooled (original 'v')",
            "rel_all": "p, vx, vy pooled (original 'combined')",
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
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
