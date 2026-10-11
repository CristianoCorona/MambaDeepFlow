"""Environment smoke test: CUDA availability and a Mamba block forward+backward.

Fails (exit code 1) if the compiled CUDA extensions `selective_scan_cuda` or
`causal_conv1d_cuda` are not importable, or if the Mamba block does not actually
call them (i.e. it would run on the slow reference/fallback path).

Usage:
    python scripts/check_env.py [--seed 0] [--batch 10] [--d-model 128]
                                [--lengths 64 256 1024] [--warmup 3] [--iters 20]
"""

import argparse
import functools
import platform
import sys


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def instrument_module(module, counter, key):
    """Wrap every public callable of an extension module with a call counter."""
    for name in dir(module):
        if name.startswith("_"):
            continue
        fn = getattr(module, name)
        if not callable(fn):
            continue

        @functools.wraps(fn)
        def wrapped(*args, __fn=fn, **kwargs):
            counter[key] += 1
            return __fn(*args, **kwargs)

        setattr(module, name, wrapped)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    import torch

    print("== Environment ==")
    print(f"python          {platform.python_version()}")
    print(f"torch           {torch.__version__}")
    print(f"torch CUDA      {torch.version.cuda}")
    print(f"cudnn           {torch.backends.cudnn.version()}")
    print(f"cxx11 ABI       {torch.compiled_with_cxx11_abi()}")
    print(f"CUDA available  {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False")
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    print(f"GPU             {props.name}, sm_{props.major}{props.minor}, "
          f"{props.total_memory / 2**20:.0f} MiB")

    # Compiled CUDA extensions: hard requirement.
    try:
        import selective_scan_cuda
    except ImportError as e:
        fail(f"selective_scan_cuda not importable: {e}")
    try:
        import causal_conv1d_cuda
    except ImportError as e:
        fail(f"causal_conv1d_cuda not importable: {e}")
    import causal_conv1d
    import mamba_ssm
    from mamba_ssm import Mamba
    from mamba_ssm.modules import mamba_simple

    print(f"mamba_ssm       {mamba_ssm.__version__}  ({mamba_ssm.__file__})")
    print(f"causal_conv1d   {causal_conv1d.__version__}  ({causal_conv1d.__file__})")
    print(f"selective_scan_cuda  {selective_scan_cuda.__file__}")
    print(f"causal_conv1d_cuda   {causal_conv1d_cuda.__file__}")

    # Count calls into the CUDA extensions and the fused fast-path entry point.
    calls = {"mamba_inner_fn": 0, "selective_scan_cuda": 0, "causal_conv1d_cuda": 0}
    instrument_module(selective_scan_cuda, calls, "selective_scan_cuda")
    instrument_module(causal_conv1d_cuda, calls, "causal_conv1d_cuda")
    inner = mamba_simple.mamba_inner_fn

    @functools.wraps(inner)
    def counted_inner(*a, **kw):
        calls["mamba_inner_fn"] += 1
        return inner(*a, **kw)

    mamba_simple.mamba_inner_fn = counted_inner

    torch.manual_seed(args.seed)
    block = Mamba(d_model=args.d_model, d_state=16, d_conv=4, expand=2).to(dev)
    n_params = sum(p.numel() for p in block.parameters())
    print(f"\nMamba block: d_model={args.d_model}, d_state=16, d_conv=4, expand=2, "
          f"params={n_params}, use_fast_path={block.use_fast_path}, seed={args.seed}")

    # Path check on one forward+backward.
    x = torch.randn(args.batch, args.lengths[0], args.d_model, device=dev, requires_grad=True)
    y = block(x)
    y.sum().backward()
    torch.cuda.synchronize()
    print(f"calls in one fwd+bwd: {calls}")
    if calls["mamba_inner_fn"] > 0 and calls["selective_scan_cuda"] > 0 \
            and calls["causal_conv1d_cuda"] > 0:
        print("path: CUDA kernels (fused mamba_inner_fn -> selective_scan_cuda + causal_conv1d_cuda)")
    else:
        print("path: fallback (reference / non-fused)")
        fail("Mamba block did not use the CUDA kernels")

    print(f"\n== Forward+backward, batch={args.batch}, fp32, "
          f"{args.warmup} warmup + {args.iters} timed iters ==")
    print(f"{'L':>6} | {'ms/iter':>9} | {'peak MiB':>9} | finite")
    for L in args.lengths:
        x = torch.randn(args.batch, L, args.d_model, device=dev, requires_grad=True)
        for _ in range(args.warmup):
            block.zero_grad(set_to_none=True)
            block(x).sum().backward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            block.zero_grad(set_to_none=True)
            y = block(x)
            y.sum().backward()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / args.iters
        peak = torch.cuda.max_memory_allocated(dev) / 2**20
        finite = bool(torch.isfinite(y).all()) and all(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in block.parameters())
        print(f"{L:>6} | {ms:>9.3f} | {peak:>9.1f} | {finite}")
        if not finite:
            fail(f"non-finite output or missing/non-finite grads at L={L}")

    print("\nOK")


if __name__ == "__main__":
    main()
