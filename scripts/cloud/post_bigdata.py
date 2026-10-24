"""Per-run post-processing on the cloud VM, after the big-dataset queue (docs/preregistration-bigdata.md).

The full set stays on the VM; these outputs travel back with sync_results.sh so that scripts/bigdata_analysis.py can run
on the laptop without the data. For every finished run under --out-root (config.json status finished, split mode
grouped-full, eval_val.json present):
  post_val.npz         explore.run_inference on the grouped-full validation (batch 1, as evaluate.py): per-sample
                       rel / abs sums, region sums (wall band, wake, rest, interior), divergence errors (DIV_MAPPING),
                       plus files and base profiles (mdata.profile_full)
  post_trainfit.json   verification 5a: train L1 and rel_* in eval() mode on the training part (per-sample sums as
                       train_fit.train_metrics, batch 10), val L1 from post_val.npz; each also for the reg/ and shear/
                       files separately (keys *_reg, *_shear; Amendment 1, 2026-10-19: main validation = reg/);
                       skipped with --no-trainfit
post_val.npz keeps the file names ("<subdir>/<name>.npz"), so wall band and divergence are split by subdirectory in
scripts/bigdata_analysis.py; post_val_subsets.json gives their means per subset (reg, shear, all) for a quick look.
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
from mdf import explore, models, runinfo
from train_fit import CH, NPIX

FIT_VERSION = 2  # 2: per-subdirectory keys (Amendment 1)
SUBSETS = {"all": "", "reg": "reg/", "shear": "shear/"}


def train_sums(run_dir, inputs, targets, batch_size):
    """Per-sample sum|o - t| and sum|t| per channel (n, 3), eval() mode, float64 (as train_fit.train_metrics)."""
    cfg = json.loads((run_dir / "config.json").read_text())
    net = models.build_from_cfg(models.model_cfg_from_run_config(cfg))
    net.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    net.cuda().eval()
    n = inputs.shape[0]
    err = torch.zeros(n, 3, dtype=torch.float64)
    tot = torch.zeros(n, 3, dtype=torch.float64)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            t = targets[s:s + batch_size].cuda().double()
            o = net(inputs[s:s + batch_size].cuda()).double()
            err[s:s + batch_size] = (o - t).abs().sum(dim=(2, 3)).cpu()
            tot[s:s + batch_size] = t.abs().sum(dim=(2, 3)).cpu()
    return err.numpy(), tot.numpy()


def fit_means(err, tot, mask, suffix):
    """L1 and rel_* means over the samples in mask (train_fit.train_metrics definitions)."""
    e, t = err[mask], tot[mask]
    out = {f"L1{suffix}": float(np.mean(e.sum(1) / NPIX)), f"rel_all{suffix}": float(np.mean(e.sum(1) / t.sum(1))),
           f"n{suffix}": int(mask.sum())}
    for i, c in enumerate(CH):
        out[f"rel_{c}{suffix}"] = float(np.mean(e[:, i] / t[:, i]))
    return out


def masks(files):
    return {k: np.array([f.startswith(pre) for f in files]) for k, pre in SUBSETS.items()}


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
        done_val = (d / "post_val.npz").exists() and (d / "post_val_subsets.json").exists() and json.loads(
            str(np.load(d / "post_val.npz")["meta"]))["model_sha256"] == sha
        fit = json.loads((d / "post_trainfit.json").read_text()) if (d / "post_trainfit.json").exists() else {}
        done_fit = args.no_trainfit or (fit.get("model_sha256") == sha and fit.get("version") == FIT_VERSION)
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
            summ = {}
            for k, m in masks(val_ds.files).items():
                wall = r["reg_err_wall"][m, 2] / np.maximum(r["reg_n_wall"][m], 1)
                summ[k] = {"n": int(m.sum()), "rel_vy": float(r["rel_vy"][m].mean()),
                           "wall_vy_mae": float(wall.mean()),
                           **{f"div_err_{g}": float(r[f"div_err_{g}"][m].mean()) for g in explore.DIV_REGIONS}}
            (d / "post_val_subsets.json").write_text(json.dumps(summ, indent=2) + "\n")
        if not done_fit:
            if train_ds is None:
                t2 = time.perf_counter()
                train_ds, _, info = mdata.load_train_val_full(subdirs=next(iter(subdirs)))
                assert info["train_files_sha256"] in train_shas, "training files differ from the runs"
                print(f"training part {len(train_ds)} samples loaded in {time.perf_counter() - t2:.0f} s")
            err, tot = train_sums(d, train_ds.inputs, train_ds.targets, 10)
            out = {"model_sha256": sha, "version": FIT_VERSION, "n_train": len(train_ds), "batch_size": 10}
            for k, m in masks(train_ds.files).items():
                out.update({f"train_{q}": v for q, v in fit_means(err, tot, m, "" if k == "all" else f"_{k}").items()})
            with np.load(d / "post_val.npz") as z:
                vfiles = [str(f) for f in z["files"]]
                for k, m in masks(vfiles).items():
                    out[f"val_L1{'' if k == 'all' else '_' + k}"] = float(np.mean(z["abs_err_all"][m] / NPIX))
            out["utc"] = runinfo.utc_now()
            (d / "post_trainfit.json").write_text(json.dumps(out, indent=2) + "\n")
        torch.cuda.empty_cache()
        print(f"{d.name}: done in {time.perf_counter() - t1:.0f} s", flush=True)


if __name__ == "__main__":
    main()
