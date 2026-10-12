"""Model cost in isolation (docs/lab-notebook.md): params, FLOPs, peak training VRAM, inference latency.

Configurations: B, B-large, CBAM8, SA8, MD8, SA32, MD32. Never run in parallel with a training run.

- params: trainable, total and added vs B (2 332 035).
- FLOPs, forward, batch 1: torch.utils.flop_counter.FlopCounterMode (matmul / conv / attention, 2 FLOPs per
  multiply-add; elementwise ops such as BatchNorm, activations, LayerNorm, softmax are not counted, for every
  model alike). For MD the fused Mamba kernels are not (fully) visible to the counter, so the MD block is counted
  analytically instead (the counter's own number for the block is also reported), per sequence of length L with
  C = d_model, E = 2C = d_inner, N = 16 = d_state, R = ceil(C / 16) = dt_rank, K = 4 = d_conv:
      in_proj 2 L C 2E + conv1d 2 L E K + x_proj 2 L E (R + 2N) + dt_proj 2 L R E + out_proj 2 L E C
      + selective scan 7 L E N   (per (l, e, n): delta*A, delta*B*u (2), h = dA*h + dBu (2), y += C*h (2))
      + D-skip and gating 3 L E
  times 4 sequences (scan directions) per image. exp / softplus / SiLU are not counted (as elsewhere).
- peak VRAM in training: batch 10, forward + backward + Adam step, 3 iterations, torch.cuda.max_memory_allocated
  (excludes the CUDA context).
- latency: eval(), no_grad, batch 1 and 10, --warmup iterations, then the median over --reps of
  synchronize -> forward -> synchronize.
- training it/s: read from the control runs (--control-root), the official training-cost measure.

Usage: python scripts/cost.py [--control-root ~/runs/control-2026-10-12] [--csv docs/results/cost.csv]
"""

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from mdf import models, runinfo

CONFIGS = {
    "B": {"variant": "B"},
    "B-large": {"variant": "B-large"},
    "CBAM8": {"variant": "CBAM", "level": 8},
    "SA8": {"variant": "SA", "level": 8},
    "MD8": {"variant": "MD", "level": 8},
    "SA32": {"variant": "SA", "level": 32},
    "MD32": {"variant": "MD", "level": 32},
}


def md_block_flops(c, size, batch=1, d_state=16, d_conv=4, expand=2):
    L, E, N, K = size * size, expand * c, d_state, d_conv
    R = math.ceil(c / 16)
    per_seq = (2 * L * c * 2 * E + 2 * L * E * K + 2 * L * E * (R + 2 * N) + 2 * L * R * E + 2 * L * E * c
               + 7 * L * E * N + 3 * L * E)
    return 4 * batch * per_seq


def flops(net, cfg):
    x = torch.randn(1, 3, 128, 128, device="cuda")
    net.eval()
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        net(x)
    total = counter.get_total_flops()
    counts = counter.get_flop_counts()
    block_counted = sum(sum(v.values()) for k, v in counts.items() if k.endswith(".mixer"))
    if cfg["variant"] == "MD":
        c = 32 * models.LEVEL_LAYER[cfg["level"]][1]
        analytic = md_block_flops(c, cfg["level"])
        return total - block_counted + analytic, block_counted, analytic
    return total, block_counted, None


def peak_train_vram(net):
    torch.cuda.empty_cache()
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=6e-4, betas=(0.5, 0.999))
    crit = nn.L1Loss()
    x = torch.randn(10, 3, 128, 128, device="cuda")
    t = torch.randn(10, 3, 128, 128, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        net.zero_grad()
        crit(net(x), t).backward()
        opt.step()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def latency_ms(net, batch, warmup, reps):
    net.eval()
    x = torch.randn(batch, 3, 128, 128, device="cuda")
    times = []
    with torch.no_grad():
        for i in range(warmup + reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            net(x)
            torch.cuda.synchronize()
            if i >= warmup:
                times.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(times)


def control_its(root, name):
    cfg = root / name / "config.json"
    if not cfg.exists():
        return None
    return json.loads(cfg.read_text()).get("it_per_s_train")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--control-root", type=Path, default=Path.home() / "runs" / "control-2026-10-12")
    ap.add_argument("--csv", type=Path, default=Path("docs/results/cost.csv"))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    git = runinfo.git_info()
    print(f"commit {git['commit']} dirty {git['dirty']}; {torch.cuda.get_device_name(0)}")

    rows = []
    for name, cfg in CONFIGS.items():
        runinfo.seed_everything(args.seed)
        net = models.build_from_cfg(cfg).cuda()
        n = models.count_params(net)
        f_total, f_block_counter, f_block_analytic = flops(net, models.normalize_cfg(cfg))
        vram = peak_train_vram(net)
        lat1 = latency_ms(net, 1, args.warmup, args.reps)
        lat10 = latency_ms(net, 10, args.warmup, args.reps)
        its = control_its(args.control_root.expanduser(), name)
        row = {"config": name, "params": n, "params_added_vs_B": n - models.B_PARAMS,
               "block_params": models.block_params(net), "gflops_fwd_b1": f_total / 1e9,
               "block_gflops_counter": f_block_counter / 1e9,
               "block_gflops_analytic": None if f_block_analytic is None else f_block_analytic / 1e9,
               "peak_train_vram_mib_b10": vram, "latency_ms_b1": lat1, "latency_ms_b10": lat10,
               "train_it_per_s_control": its, "commit": git["commit"]}
        rows.append(row)
        print(f"{name:8s} params {n:9d} ({n - models.B_PARAMS:+8d})  {f_total / 1e9:7.4f} GFLOPs  "
              f"VRAM {vram:7.1f} MiB  lat b1 {lat1:6.2f} ms  b10 {lat10:6.2f} ms  it/s {its}", flush=True)
        del net
        torch.cuda.empty_cache()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
