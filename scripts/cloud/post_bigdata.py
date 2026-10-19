"""Per-run post-processing on the cloud VM, after the big-dataset queue (docs/preregistration-bigdata.md).

The full set stays on the VM; these outputs travel back with sync_results.sh so that scripts/bigdata_analysis.py can run
on the laptop without the data. For every finished run under --out-root (config.json status finished, split mode
grouped-full, eval_val.json present):
  post_val.npz         explore.run_inference on the grouped-full validation (batch 1, as evaluate.py): per-sample
                       rel / abs sums, region sums (wall band, wake, rest, interior), divergence errors (DIV_MAPPING),
                       plus files and base profiles (mdata.profile_full)
  post_trainfit.json   verification 5a: train L1 and rel_* in eval() mode on the training part (train_fit.train_metrics,
                       batch 10), val L1 from post_val.npz; skipped with --no-trainfit
A run whose outputs exist with the same model.pt sha256 is skipped. The test set is not on the VM and never read.

Usage:
    python scripts/cloud/post_bigdata.py --out-root ~/runs/bigdata [--no-trainfit]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root, for `mdf`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/, for train_fit

import numpy as np
import torch

from mdf import data as mdata
from mdf import explore, runinfo
from train_fit import NPIX, train_metrics


def finished_runs(out_root):
    runs = []
    for cfg_path in sorted(Path(out_root).glob("*/config.json")):
        cfg = json.loads(cfg_path.read_text())
        d = cfg_path.parent
        if cfg.get("status") == "finished" and cfg.get("split", {}).get("split_mode") == mdata.GROUPED_FULL \
                and (d / "eval_val.json").exists():
            runs.append((d, cfg))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--no-trainfit", action="store_true")
    args = ap.parse_args()
    runs = finished_runs(args.out_root)
    if not runs:
        sys.exit(f"no finished grouped-full runs under {args.out_root}")
    subdirs = {tuple(cfg["split"]["full_subdirs"]) for _, cfg in runs}
    val_shas = {cfg["split"]["val_files_sha256"] for _, cfg in runs}
    train_shas = {cfg["split"]["train_files_sha256"] for _, cfg in runs}
    if len(subdirs) != 1 or len(val_shas) != 1 or len(train_shas) != 1:
        sys.exit(f"runs differ in subdirs / validation / training files: {subdirs}, {val_shas}, {train_shas}")
    todo = []
    for d, cfg in runs:
        sha = explore.file_sha256(d / "model.pt")
        done_val = (d / "post_val.npz").exists() and json.loads(str(np.load(d / "post_val.npz")["meta"]))[
            "model_sha256"] == sha
        done_fit = args.no_trainfit or ((d / "post_trainfit.json").exists() and json.loads(
            (d / "post_trainfit.json").read_text())["model_sha256"] == sha)
        if not (done_val and done_fit):
            todo.append((d, cfg, sha, done_val, done_fit))
    print(f"{len(runs)} finished runs, {len(todo)} to process")
    if not todo:
        return
    t0 = time.perf_counter()
    val_ds = mdata.load_val_full(subdirs=next(iter(subdirs)))
    assert mdata.file_list_sha256(val_ds.files) in val_shas, "validation files differ from the runs"
    profiles = [mdata.profile_full(f) for f in val_ds.files]
    props = explore.sample_properties(val_ds)
    regions = explore.all_region_masks(val_ds, props)
    print(f"validation {len(val_ds)} samples ({len(set(profiles))} profiles) in {time.perf_counter() - t0:.0f} s")
    train_ds = None
    for d, cfg, sha, done_val, done_fit in todo:
        t1 = time.perf_counter()
        if not done_val:
            r = explore.run_inference(d, val_ds, regions, [])
            meta = {"model_sha256": sha, "val_files_sha256": mdata.file_list_sha256(val_ds.files),
                    "div_mapping": list(explore.DIV_MAPPING), "wall_px": explore.WALL_PX,
                    "wake_half_width": explore.WAKE_HALF_WIDTH, "git": runinfo.git_info(), "utc": runinfo.utc_now()}
            np.savez_compressed(d / "post_val.npz", meta=json.dumps(meta), files=np.array(val_ds.files),
                                profiles=np.array(profiles), **{k: v for k, v in r.items() if k != "keep_pred"})
        if not done_fit:
            if train_ds is None:
                t2 = time.perf_counter()
                train_ds, _, info = mdata.load_train_val_full(subdirs=next(iter(subdirs)))
                assert info["train_files_sha256"] in train_shas, "training files differ from the runs"
                print(f"training part {len(train_ds)} samples loaded in {time.perf_counter() - t2:.0f} s")
            tr = train_metrics(d, train_ds.inputs, train_ds.targets, 10)
            with np.load(d / "post_val.npz") as z:
                val_l1 = float(np.mean(z["abs_err_all"] / NPIX))
            (d / "post_trainfit.json").write_text(json.dumps(
                {"model_sha256": sha, "n_train": len(train_ds), "batch_size": 10,
                 **{f"train_{k}": v for k, v in tr.items()}, "val_L1": val_l1, "utc": runinfo.utc_now()}, indent=2)
                + "\n")
        torch.cuda.empty_cache()
        print(f"{d.name}: done in {time.perf_counter() - t1:.0f} s", flush=True)


if __name__ == "__main__":
    main()
