"""Block-level micro-benchmark: time and memory vs number of tokens (docs/lab-notebook.md, D12).

Blocks (d = 64 channels, batch 10, square grids 8..128 = 64..16384 tokens):
  MD         SS2DMamba (4 scans, fused CUDA kernels), out_proj randomised (with zeros the cost is the same,
             but the gradients upstream would be zero)
  SA-SDPA    SelfAttention2D (F.scaled_dot_product_attention; the backend PyTorch selects is recorded)
  SA-naive   same block with an explicit attention matrix softmax(QK^T / sqrt(d)) V
Measured: forward + backward time (median over --reps after --warmup, torch.cuda.synchronize around each rep)
and peak allocated memory during one forward + backward, minus the memory held before it (parameters, input).
An out-of-memory error is a result ("OOM"), not a failure. Never run in parallel with a training run.

Outputs: CSV (--csv) and a log-log PNG (--png) with time and memory vs tokens.

Usage: python scripts/microbench.py [--grids 8 16 32 64 128] [--reps 20] [--warmup 5]
                                    [--csv docs/results/microbench.csv] [--png docs/figures/microbench.png]
"""

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from mdf import blocks, runinfo

D, BATCH = 64, 10
COLORS = {"MD": "#2a78d6", "SA-SDPA": "#eb6834", "SA-naive": "#1baf7a"}  # categorical slots 1-3


class NaiveSelfAttention2D(blocks.SelfAttention2D):
    def forward(self, x):
        b, c, h, w = x.shape
        t = self.norm(x.flatten(2).transpose(1, 2)) + self.pe
        q, k, v = self.qkv(t).view(b, h * w, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)
        a = (attn @ v).transpose(1, 2).reshape(b, h * w, c)
        return x + self.out(a).transpose(1, 2).reshape(b, c, h, w)


def make(kind, grid):
    if kind == "MD":
        blk = blocks.SS2DMamba(D, "cuda")
        nn.init.normal_(blk.mamba.out_proj.weight, std=0.02)
    elif kind == "SA-SDPA":
        blk = blocks.SelfAttention2D(D, grid, grid)
    else:
        blk = NaiveSelfAttention2D(D, grid, grid)
    if kind != "MD":
        nn.init.normal_(blk.out.weight, std=0.02)
    return blk.cuda()


def sdpa_backend(grid):
    """Backend PyTorch picks for the SA-SDPA shapes (float32, no mask)."""
    heads = 4
    q = torch.randn(BATCH, heads, grid * grid, D // heads, device="cuda")
    try:
        choice = torch._fused_sdp_choice(q, q, q)
        return torch.nn.attention.SDPBackend(choice).name
    except Exception as e:  # private API; record why it is unknown
        return f"unknown ({type(e).__name__})"


def bench(kind, grid, reps, warmup):
    torch.cuda.empty_cache()
    blk = make(kind, grid)
    x = torch.randn(BATCH, D, grid, grid, device="cuda", requires_grad=True)

    def step():
        y = blk(x)
        y.sum().backward()
        blk.zero_grad(set_to_none=True)
        x.grad = None

    try:
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        times = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return {"time_ms": 1e3 * statistics.median(times), "time_ms_min": 1e3 * min(times),
                "peak_mib": peak / 2**20, "status": "ok"}
    except torch.cuda.OutOfMemoryError:
        return {"time_ms": None, "time_ms_min": None, "peak_mib": None, "status": "OOM"}
    finally:
        del blk, x
        torch.cuda.empty_cache()


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in ((axes[0], "time_ms", "fwd + bwd time (ms, median)"),
                           (axes[1], "peak_mib", "peak extra memory (MiB)")):
        for kind, color in COLORS.items():
            pts = [(r["tokens"], r[key]) for r in rows if r["block"] == kind and r["status"] == "ok"]
            oom = [r["tokens"] for r in rows if r["block"] == kind and r["status"] == "OOM"]
            lab = kind + (f" (OOM at {', '.join(map(str, oom))} tokens)" if oom else "")
            if pts:
                ax.plot(*zip(*pts), marker="o", markersize=5, linewidth=2, color=color, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("tokens (grid x grid)")
        ax.set_ylabel(label)
        ax.grid(True, which="major", color="#dddddd", linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Block micro-benchmark, d = {D}, batch {BATCH}, float32, {torch.cuda.get_device_name(0)}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grids", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=Path, default=Path("docs/results/microbench.csv"))
    ap.add_argument("--png", type=Path, default=Path("docs/figures/microbench.png"))
    args = ap.parse_args()
    runinfo.seed_everything(args.seed)
    git = runinfo.git_info()
    print(f"commit {git['commit']} dirty {git['dirty']}; {torch.cuda.get_device_name(0)}")

    rows = []
    for grid in args.grids:
        backend = sdpa_backend(grid)
        for kind in COLORS:
            r = bench(kind, grid, args.reps, args.warmup)
            r.update({"block": kind, "grid": grid, "tokens": grid * grid, "d": D, "batch": BATCH,
                      "sdpa_backend": backend if kind == "SA-SDPA" else "",
                      "commit": git["commit"]})
            rows.append(r)
            t = "OOM" if r["status"] == "OOM" else f"{r['time_ms']:.2f} ms, {r['peak_mib']:.1f} MiB"
            print(f"grid {grid:3d} ({grid * grid:5d} tokens) {kind:8s} {t}" + (f"  [{backend}]" if r["sdpa_backend"] else ""),
                  flush=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["block", "grid", "tokens", "d", "batch", "time_ms", "time_ms_min",
                                           "peak_mib", "status", "sdpa_backend", "commit"])
        w.writeheader()
        w.writerows(rows)
    plot(rows, args.png)
    print(f"Wrote {args.csv}, {args.png}")


if __name__ == "__main__":
    main()
