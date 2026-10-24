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

Set "final" (--set final; added 2026-10-18 for docs/preregistration-final.md): B, MDv64, MDv64-local, SA64, same
measurements and conditions. Additions:
- MDv / MDv-local (SS2DVMamba, mdf/blocks.py; SS2DVMambaLocal, mdf/ablation.py): the counter sees the linear layers,
  the depthwise conv and the x_proj / dt_proj einsums but not the selective scan, so the block is counted analytically
  (the counter's own number is also reported), per image with L = H W, C, E = 2C, N = 16, R = ceil(C / 16), 3x3
  depthwise conv, 4 scan directions:
      in_proj 2 L C 2E + conv2d 2 L E 9
      + per direction: x_proj 2 L E (R + 2N) + dt_proj 2 L R E + selective scan 7 L E N + D-skip 2 L E
      + sum of the 4 directions 3 L E + gating L E + out_proj 2 L E C
  MDv-local: the same without x_proj, dt_proj and the scan (its forward_core is sum_k D_k x: the D-skip terms remain).
  LayerNorms / exp / softplus / SiLU are not counted (as elsewhere).
- params_used: parameters the forward uses (MDv-local: total minus x_proj / dt_proj / A_logs, mdf.ablation).
- training it/s: mean (min-max) of it_per_s_train over the existing 133-epoch runs of each configuration (FINAL_RUNS),
  instead of the control runs.
- also writes a markdown table (--md).

Usage: python scripts/cost.py [--control-root ~/runs/control-2026-10-12] [--csv docs/results/cost.csv]
       python scripts/cost.py --set final --csv docs/results/cost-final/cost.csv --md docs/results/cost-final/cost.md
"""

import argparse
import csv
import glob
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
# final architecture and its references (docs/preregistration-final.md, 2026-10-18; --set final)
FINAL_CONFIGS = {
    "B": {"variant": "B"},
    "MDv64": {"variant": "MDv", "level": 64},
    "MDv64-local": {"variant": "MDv-local", "level": 64},
    "SA64": {"variant": "SA", "level": 64},
}
FINAL_RUNS = {  # existing 133-epoch runs (under ~/runs), for the training it/s
    "B": ("night1/ep133_s*", "night3/B_s*"),
    "MDv64": ("night5/MDv64_s*", "night7/MDv64_s*"),
    "MDv64-local": ("night8a/MDv64-local_s*",),
    "SA64": ("night9/SA64_s*",),
}


def md_block_flops(c, size, batch=1, d_state=16, d_conv=4, expand=2):
    L, E, N, K = size * size, expand * c, d_state, d_conv
    R = math.ceil(c / 16)
    per_seq = (2 * L * c * 2 * E + 2 * L * E * K + 2 * L * E * (R + 2 * N) + 2 * L * R * E + 2 * L * E * c
               + 7 * L * E * N + 3 * L * E)
    return 4 * batch * per_seq


def mdv_block_flops(c, size, batch=1, d_state=16, d_conv=3, expand=2, local=False):
    L, E, N, K = size * size, expand * c, d_state, 4
    R = math.ceil(c / 16)
    per_dir = 2 * L * E if local else 2 * L * E * (R + 2 * N) + 2 * L * R * E + 7 * L * E * N + 2 * L * E
    return batch * (2 * L * c * 2 * E + 2 * L * E * d_conv ** 2 + K * per_dir + 3 * L * E + L * E + 2 * L * E * c)


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
    if cfg["variant"] in ("MDv", "MDv-local"):
        c = 32 * models.LEVEL_LAYER[cfg["level"]][1]
        analytic = mdv_block_flops(c, cfg["level"], local=cfg["variant"] == "MDv-local")
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


def used_params(net):
    from mdf import ablation
    mixer = getattr(net, "mixer", None)
    if mixer is None:
        return models.count_params(net)
    return models.count_params(net) - models.count_params(mixer) + ablation.used_param_count(mixer)


def runs_its(label):
    vals = []
    for pat in FINAL_RUNS[label]:
        for cfg in sorted(glob.glob(str(Path.home() / "runs" / pat / "config.json"))):
            c = json.loads(Path(cfg).read_text())
            if c.get("status") == "finished" and c.get("args", {}).get("epochs") == 133:
                vals.append(c["it_per_s_train"])
    return vals


def write_md(path, rows):
    head = ("| Config | Params total | Params used | Added vs B (used) | GFLOPs fwd (b=1) | Block GFLOPs (counter / "
            "analytic) | Peak train VRAM b10 (MiB) | Latency b1 / b10 (ms) | Train it/s, runs: mean (min-max, n) |")
    out = [head, "|" + "---|" * 9]
    for r in rows:
        an = r["block_gflops_analytic"]
        th = lambda n, sign="": f"{n:{sign},}".replace(",", " ")
        out.append(f"| {r['config']} | {th(r['params'])} | {th(r['params_used'])} | "
                   f"{th(r['params_used'] - models.B_PARAMS, '+')} | "
                   f"{r['gflops_fwd_b1']:.4f} | {r['block_gflops_counter']:.4f} / "
                   f"{'-' if an is None else f'{an:.4f}'} | {r['peak_train_vram_mib_b10']:.1f} | "
                   f"{r['latency_ms_b1']:.2f} / {r['latency_ms_b10']:.2f} | {r['train_it_per_s_runs']:.1f} "
                   f"({r['train_it_per_s_runs_min']:.1f}-{r['train_it_per_s_runs_max']:.1f}, {r['train_runs']}) |")
    path.write_text("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--control-root", type=Path, default=Path.home() / "runs" / "control-2026-10-12")
    ap.add_argument("--csv", type=Path, default=Path("docs/results/cost.csv"))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", choices=["main", "final"], default="main",
                    help="main: the 13/10 configurations (default); final: B, MDv64, MDv64-local, SA64")
    ap.add_argument("--md", type=Path, default=None, help="also write a markdown table (--set final)")
    args = ap.parse_args()
    final = args.set == "final"
    git = runinfo.git_info()
    print(f"commit {git['commit']} dirty {git['dirty']}; {torch.cuda.get_device_name(0)}")

    rows = []
    for name, cfg in (FINAL_CONFIGS if final else CONFIGS).items():
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
        if final:
            vals = runs_its(name)
            row = {"config": name, "params": n, "params_used": used_params(net), **{k: v for k, v in row.items()
                   if k not in ("config", "params", "train_it_per_s_control", "commit")},
                   "train_it_per_s_runs": statistics.mean(vals), "train_it_per_s_runs_min": min(vals),
                   "train_it_per_s_runs_max": max(vals), "train_runs": len(vals), "commit": git["commit"]}
            its = f"{row['train_it_per_s_runs']:.1f} (runs, n={len(vals)}); params used {row['params_used']}"
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
    if args.md is not None:
        write_md(args.md, rows)
        print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
