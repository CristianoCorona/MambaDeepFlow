"""Controlled training run (baseline and future variants share this script and protocol).

Training loop, optimiser, LR decay and loss are those of train/runTrain.py; deviations
(fixed split, float32 loading, seeding, logging) are declared in docs/lab-notebook.md (D1-D7).

Outputs in the run directory (default ~/runs/<run-id>/, outside the repo):
  config.json  configuration, seed, split hash, git commit + dirty flag, versions, GPU,
               timings, peak RAM / VRAM
  metrics.csv  one row per epoch (definitions below)
  model.pt     final state_dict (same format as the original modelG)

metrics.csv columns:
  epoch                      0-based, as printed by the original
  iteration_end              optimiser steps done at the end of the epoch
  lr                         learning rate used in this epoch
  train_L1_mean_of_batches   mean of the per-batch L1 loss over all training batches of the epoch
                             (weights updated during the epoch; same as L1_accum in the original)
  val_L1_mean_of_batches     mean of the per-batch L1 loss over all validation batches (40 x 10 samples),
                             model in eval() mode after the epoch (same as L1val_accum in the original)
  n_train_batches, n_val_batches, train_seconds, val_seconds

Throughput: it_per_s_train = optimiser steps / time of the training loops only (includes DataLoader,
host-to-device copies, forward, backward, optimiser step, loss .item(); excludes validation, checkpointing
and data loading). it_per_s_incl_val additionally includes the validation passes in the denominator.

Usage:
    python scripts/train.py --seed 0 [--epochs 133 | --iterations 10000]
                            [--split-mode {grouped,random,kfold} [--n-folds 5 --fold K]]
                            [--variant {B,B-large,CBAM,SA,MD}] [--level {64,32,16,8}] [--channels N]
                            [--mamba-impl {cuda,ref}] [--lr 6e-4] [--batch-size 10] [--out-dir DIR]
                            [--deterministic]

The model is built by mdf.models.build_from_cfg (the single factory, also used by evaluate.py).
--model dfpnet / --expo E are kept only to reproduce earlier commands; they mean variant B with 2**E channels.
The evaluated checkpoint is always the final-epoch model (no selection on validation).

The protocol is defined in epochs (docs/lab-notebook.md, 2026-10-11): --epochs takes precedence over
--iterations, which is kept to reproduce earlier runs. Default split: profile-disjoint ("grouped").
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `mdf`

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mdf import data as mdata
from mdf import models, runinfo

CSV_COLUMNS = ["epoch", "iteration_end", "lr", "train_L1_mean_of_batches", "val_L1_mean_of_batches",
               "n_train_batches", "n_val_batches", "train_seconds", "val_seconds"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, required=True, help="weights init + batch order (not the split)")
    ap.add_argument("--iterations", type=int, default=10000,
                    help="requested; epochs = int(iterations / batches_per_epoch + 0.5) as in the original")
    ap.add_argument("--epochs", type=int, default=None, help="number of epochs; overrides --iterations")
    ap.add_argument("--split-mode", default="grouped", choices=sorted(mdata.ALL_SPLIT_MODES),
                    help="grouped: profile-disjoint validation (default); random: original-style split (D1); "
                         "kfold: profile-disjoint folds of data/train (needs --n-folds, --fold)")
    ap.add_argument("--n-folds", type=int, default=None, help="kfold only: number of folds (e.g. 5)")
    ap.add_argument("--fold", type=int, default=None, help="kfold only: validation fold, 0..n_folds-1")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--variant", default=None, choices=models.VARIANTS, help="model variant (default B)")
    ap.add_argument("--level", type=int, default=None, choices=models.LEVELS,
                    help="feature-map size where the block sits (default 8; ignored for B, B-large)")
    ap.add_argument("--channels", type=int, default=None, help="base channels (default 32; B-large: 33)")
    ap.add_argument("--mamba-impl", default=None, choices=["cuda", "ref"], help="MD only (default cuda)")
    ap.add_argument("--expo", type=int, default=None, help="legacy: B with 2**expo base channels")
    ap.add_argument("--model", default=None, choices=sorted(models.MODELS), help="legacy alias of variant B")
    ap.add_argument("--no-lr-decay", action="store_true", help="disable the original LR decay (default: on)")
    ap.add_argument("--out-dir", type=Path, default=None, help="run directory (default ~/runs/<run-id>)")
    ap.add_argument("--deterministic", action="store_true", help="cudnn.deterministic (off in the original)")
    args = ap.parse_args()
    if (args.split_mode == mdata.KFOLD) != (args.n_folds is not None and args.fold is not None):
        ap.error("--n-folds and --fold are required with --split-mode kfold and only allowed with it")
    if (args.model or args.expo is not None) and args.variant not in (None, "B"):
        ap.error("--model / --expo are legacy options for variant B only")
    if args.expo is not None and args.channels is not None:
        ap.error("give --channels or --expo, not both")
    return args


def model_cfg_from_args(args):
    channels = args.channels if args.expo is None else int(2 ** args.expo + 0.5)
    return models.normalize_cfg({"variant": args.variant or "B", "level": args.level, "channels": channels,
                                 "mamba_impl": args.mamba_impl})


def main():
    args = parse_args()
    mcfg = model_cfg_from_args(args)
    t_start = time.perf_counter()
    timers = runinfo.Timers()
    started = runinfo.utc_now()

    budget = f"ep{args.epochs}" if args.epochs is not None else f"it{args.iterations}"
    tag = mcfg["variant"] + (f"{mcfg['level']}" if mcfg["level"] else "") + f"_c{mcfg['channels']}"
    split_tag = args.split_mode + (f"{args.n_folds}f{args.fold}" if args.split_mode == mdata.KFOLD else "")
    run_id = f"{tag}_{budget}_{split_tag}_s{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = args.out_dir or Path.home() / "runs" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        sys.exit(f"Run directory {run_dir} exists and is not empty")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    if not torch.cuda.is_available():
        sys.exit("CUDA not available")

    loader_gen = runinfo.seed_everything(args.seed, args.deterministic)

    with timers("data_load"):
        train_ds, val_ds, split_info = mdata.load_train_val(
            split_file=mdata.split_file_path(args.split_mode, n_folds=args.n_folds, fold=args.fold),
            split_mode=args.split_mode, n_folds=args.n_folds, fold=args.fold)
    print(f"Train / validation samples: {len(train_ds)} / {len(val_ds)}; "
          f"val sha256 {split_info['val_files_sha256']}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              generator=loader_gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)
    if args.epochs is not None:
        epochs = args.epochs
    else:
        epochs = int(args.iterations / len(train_loader) + 0.5)
    effective_iterations = epochs * len(train_loader)
    print(f"Batches train / val: {len(train_loader)} / {len(val_loader)}; "
          f"epochs {epochs}, effective iterations {effective_iterations}")

    with timers("setup"):
        net = models.build_from_cfg(mcfg)
        n_params = models.count_params(net)
        n_block = models.block_params(net)
        net.cuda()
        criterion = nn.L1Loss().cuda()
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=0.0)
    print(f"Model {mcfg}: {n_params} trainable params ({n_params - models.B_PARAMS:+d} vs B, block {n_block})")

    config = {
        "run_id": run_dir.name,
        "status": "running",
        "started_utc": started,
        "command": runinfo.command_line(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "seed": args.seed,
        "model": mcfg["variant"],
        "variant": mcfg["variant"],
        "level": mcfg["level"],
        "channels": mcfg["channels"],
        "mamba_impl": mcfg["mamba_impl"],
        "trainable_params": n_params,
        "block_params": n_block,
        "params_B_reference": models.B_PARAMS,
        "params_added_vs_B": n_params - models.B_PARAMS,
        "evaluated_checkpoint": "final epoch (model.pt); validation is for monitoring only",
        "optimizer": {"name": "Adam", "lr": args.lr, "betas": [0.5, 0.999], "weight_decay": 0.0},
        "lr_decay": None if args.no_lr_decay else "train/utils.py computeLR(epoch, epochs, 0.1*lr, lr)",
        "loss": "nn.L1Loss (mean)",
        "batch_size": args.batch_size,
        "requested_iterations": None if args.epochs is not None else args.iterations,
        "requested_epochs": args.epochs,
        "split_mode": args.split_mode,
        "epochs": epochs,
        "batches_per_epoch": len(train_loader),
        "effective_iterations": effective_iterations,
        "split": split_info,
        "normalization": {"max_inputs": mdata.MAX_INPUTS, "max_targets": mdata.MAX_TARGETS,
                          "remove_p_offset": True, "dimensionless": True, "dtype": "float32"},
        "dataloader": {"shuffle": True, "drop_last": True, "num_workers": 0, "generator_seed": args.seed},
        "git": runinfo.git_info(),
        "env": runinfo.env_info(),
        "metrics_csv_definitions": {
            "train_L1_mean_of_batches": "mean per-batch L1 over all training batches of the epoch",
            "val_L1_mean_of_batches": "mean per-batch L1 over all validation batches, eval() mode",
        },
        "it_per_s_definition": ("optimiser steps / training-loop wall time (DataLoader, H2D copy, fwd, bwd, "
                                "step, loss.item(); excludes validation, checkpoint, data loading); "
                                "CUDA synchronised at loop boundaries"),
    }
    runinfo.write_json(run_dir / "config.json", config)

    csv_fh = open(run_dir / "metrics.csv", "w", newline="")
    writer = csv.writer(csv_fh)
    writer.writerow(CSV_COLUMNS)

    iteration = 0
    for epoch in range(epochs):
        # LR decay as in the original: computed per epoch, applied only below the base LR
        curr_lr = args.lr
        if not args.no_lr_decay:
            lr_e = models.compute_lr(epoch, epochs, args.lr * 0.1, args.lr)
            if lr_e < args.lr:
                curr_lr = lr_e
                for g in optimizer.param_groups:
                    g["lr"] = lr_e

        t0 = time.perf_counter()
        with timers("train_loop"):
            net.train()
            l1_accum = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                net.zero_grad()
                out = net(inputs)
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()
                l1_accum += loss.item()
                iteration += 1
        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        with timers("validation"):
            net.eval()
            l1val_accum = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    l1val_accum += criterion(net(inputs), targets).item()
        t_val = time.perf_counter() - t0

        train_l1 = l1_accum / len(train_loader)
        val_l1 = l1val_accum / len(val_loader)
        writer.writerow([epoch, iteration, f"{curr_lr:.10g}", f"{train_l1:.10g}", f"{val_l1:.10g}",
                         len(train_loader), len(val_loader), f"{t_train:.3f}", f"{t_val:.3f}"])
        csv_fh.flush()
        print(f"Epoch {epoch + 1}/{epochs}  it {iteration}  lr {curr_lr:.3g}  "
              f"train L1 {train_l1:.6f}  val L1 {val_l1:.6f}  ({t_train:.1f}s + {t_val:.1f}s)", flush=True)
    csv_fh.close()

    with timers("checkpoint"):
        torch.save(net.state_dict(), run_dir / "model.pt")

    total = time.perf_counter() - t_start
    t = timers.totals
    config.update({
        "status": "finished",
        "finished_utc": runinfo.utc_now(),
        "final_train_L1_mean_of_batches": train_l1,
        "final_val_L1_mean_of_batches": val_l1,
        "timing_seconds": {
            "data_load": t["data_load"],
            "setup": t["setup"],
            "train_loop": t["train_loop"],
            "validation": t["validation"],
            "checkpoint": t["checkpoint"],
            "validation_plus_checkpoint": t["validation"] + t["checkpoint"],
            "total_in_script": total,
        },
        "it_per_s_train": iteration / t["train_loop"],
        "it_per_s_incl_val": iteration / (t["train_loop"] + t["validation"]),
        "peak_rss_mib": runinfo.peak_rss_mib(),
        "peak_vram": runinfo.peak_vram(),
    })
    runinfo.write_json(run_dir / "config.json", config)
    print(f"Done: {iteration} iterations, {config['it_per_s_train']:.1f} it/s (train loop), "
          f"total {total:.1f}s, peak RSS {config['peak_rss_mib']:.0f} MiB, "
          f"peak VRAM alloc {config['peak_vram']['max_memory_allocated_mib']:.0f} MiB")


if __name__ == "__main__":
    main()
