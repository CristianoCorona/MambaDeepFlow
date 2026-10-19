"""Per-tensor report of the initial weights of the Mamba variants, to explain fingerprint differences between machines.

tests/test_multilevel.py::test_existing_variants_unchanged compares a sha256 of the whole initial state dict (seed 0,
CPU) with values computed on the laptop. On the cloud VM it fails only for MD / MDv. Hypothesis (2026-10-19): the block
initialisation uses transcendental functions (exp / log / expm1 for the dt bias and A_log, erfinv inside
trunc_normal_), whose vectorised CPU kernels can differ in the last bit between CPUs / instruction sets.

For MD8, MDv8 and MDv64 (default; --all adds MD32, MDv32, MD64), built exactly as in the test (tests/test_blocks.build:
torch.manual_seed(0), models.build_from_cfg, CPU), prints a header (CPU model and flags, torch, ATen CPU capability,
threads, mamba_ssm version, path and sha256 of its Python sources) and one line per tensor:
    T <model> <name> <sha256[:16]> <sum, float64, 17 digits> <max |x|> <shape>
plus the whole-model fingerprint (same function as the test) next to the expected one.
--npz FILE saves the tensors that are not in B (the block parameters) for an element-wise comparison.
--sweep re-runs the model fingerprints with ATEN_CPU_CAPABILITY=default / avx2 / avx512 (subprocesses).

Compare two machines:
    python scripts/cloud/fingerprint_report.py --compare laptop.txt vm.txt [--npz-a laptop.npz --npz-b vm.npz]
lists every tensor whose hash differs, with the relative difference of the sums and, with the npz files, the max
absolute difference and the max distance in float32 ULPs; and whether the mamba_ssm sources are identical.

Usage:
    python scripts/cloud/fingerprint_report.py [--all] [--npz FILE] [--sweep] > report.txt
"""

import argparse
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

MODELS = [("MD", 8), ("MDv", 8), ("MDv", 64)]
MODELS_ALL = MODELS + [("MD", 32), ("MDv", 32), ("MD", 64)]
FLAGS = ("sse4_2", "avx", "avx2", "fma", "avx512f", "avx512dq", "avx512bw", "avx512vl", "avx512_bf16", "amx_tile")


def cpu_info():
    model, flags = "?", set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") and model == "?":
                model = line.split(":", 1)[1].strip()
            if line.startswith("flags") and not flags:
                flags = set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return model, [f for f in FLAGS if f in flags]


def sources_sha256(pkg_dir):
    h = hashlib.sha256()
    for p in sorted(Path(pkg_dir).rglob("*.py")):
        h.update(str(p.relative_to(pkg_dir)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def report(args):
    import numpy as np
    import torch
    from test_blocks import build
    from test_multilevel import FINGERPRINTS, fingerprint
    b_keys = set(build("B", None).state_dict())  # before importing mamba_ssm directly: loads train/utils.py first
    import mamba_ssm

    model, flags = cpu_info()
    pkg = Path(mamba_ssm.__file__).parent
    print(f"H cpu_model {model}")
    print(f"H cpu_flags {' '.join(flags)}")
    print(f"H nproc {os.cpu_count()} torch_threads {torch.get_num_threads()}")
    print(f"H python {platform.python_version()} torch {torch.__version__} "
          f"aten_cpu_capability {torch.backends.cpu.get_cpu_capability()} "
          f"ATEN_CPU_CAPABILITY_env {os.environ.get('ATEN_CPU_CAPABILITY', '')!r}")
    print(f"H mamba_ssm {mamba_ssm.__version__} path {str(pkg).replace(str(Path.home()), '~')} sources_sha256 {sources_sha256(pkg)}")
    print(f"H repo_commit {subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()}")
    saved = {}
    for variant, level in (MODELS_ALL if args.all else MODELS):
        net = build(variant, level)
        name = f"{variant}{level}"
        fp, exp = fingerprint(net), FINGERPRINTS[(variant, level)]
        print(f"M {name} fingerprint {fp} expected {exp} {'MATCH' if fp == exp else 'DIFFERS'}")
        for k, v in net.state_dict().items():
            v64 = v.detach().double()
            print(f"T {name} {k} {tensor_sha(v)} {float(v64.sum()):.17g} {float(v64.abs().max()) if v.numel() else 0:.9g} "
                  f"{'x'.join(map(str, v.shape)) or 'scalar'}")
            if k not in b_keys:
                saved[f"{name}/{k}"] = v.detach().cpu().numpy()
    if args.npz:
        np.savez_compressed(args.npz, **saved)
        print(f"H npz {args.npz} ({len(saved)} block tensors)")


def sweep():
    code = ("import sys; sys.path[:0] = [%r, %r]; import torch; from test_blocks import build; "
            "from test_multilevel import fingerprint; "
            "print(torch.backends.cpu.get_cpu_capability(), ' '.join(f'{v}{l}:{fingerprint(build(v, l))}' "
            "for v, l in %r))" % (str(REPO), str(REPO / "tests"), MODELS_ALL))
    for cap in ("default", "avx2", "avx512"):
        out = subprocess.run([sys.executable, "-c", code], env={**os.environ, "ATEN_CPU_CAPABILITY": cap},
                             capture_output=True, text=True, cwd=REPO)
        print(f"S requested {cap}: {(out.stdout.strip() or out.stderr.strip().splitlines()[-1])}")


def parse(path):
    head, models_, tens = {}, {}, {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "H":
            head[parts[1]] = " ".join(parts[2:])
        elif parts[0] == "M":
            models_[parts[1]] = parts[3]
        elif parts[0] == "T":
            tens[(parts[1], parts[2])] = (parts[3], float(parts[4]), float(parts[5]), parts[6])
    return head, models_, tens


def compare(args):
    import numpy as np
    ha, ma, ta = parse(args.compare[0])
    hb, mb, tb = parse(args.compare[1])
    for k in ("cpu_model", "cpu_flags", "python", "mamba_ssm", "nproc"):
        print(f"{k}:\n  A {ha.get(k)}\n  B {hb.get(k)}")
    sa = ha.get("mamba_ssm", "").split("sources_sha256 ")[-1]
    sb = hb.get("mamba_ssm", "").split("sources_sha256 ")[-1]
    print(f"mamba_ssm Python sources identical: {sa == sb} ({sa} vs {sb})")
    for m in ma:
        print(f"model {m}: A {ma[m]}  B {mb.get(m)}  {'same' if ma[m] == mb.get(m) else 'DIFFERENT'}")
    za = np.load(args.npz_a) if args.npz_a else None
    zb = np.load(args.npz_b) if args.npz_b else None
    diff = [k for k in ta if k in tb and ta[k][0] != tb[k][0]]
    print(f"\n{len(diff)} of {len(ta)} tensors differ (hash):")
    for k in diff:
        (_, s1, m1, shape), (_, s2, m2, _) = ta[k], tb[k]
        rel = abs(s1 - s2) / max(abs(s1), 1e-300)
        line = f"  {k[0]:6s} {k[1]:45s} {shape:>12s}  sum rel diff {rel:.2e}  max|x| {m1:.6g} / {m2:.6g}"
        key = f"{k[0]}/{k[1]}"
        if za is not None and zb is not None and key in za.files and key in zb.files:
            a, b = za[key], zb[key]
            ulp = np.abs(a.astype(np.float32).view(np.int32).astype(np.int64)
                         - b.astype(np.float32).view(np.int32).astype(np.int64))
            rel = float(np.abs(a - b).max()) / max(float(np.abs(a).max()), 1e-300)
            line += (f"  elements differing {int((a != b).sum())}/{a.size}  max|a-b| {float(np.abs(a - b).max()):.3g}"
                     f" ({rel:.2e} of max|x|)  max ULP {int(ulp.max())} (large near 0)")
        print(line)
    in_block = [k for k in diff if ".mixer" in "." + k[1] or k[1].startswith("mixer")]
    print(f"\nSummary: {len(diff)} differing tensors, {len(in_block)} in the block (mixer.*), "
          f"{len(diff) - len(in_block)} in the B skeleton: {sorted({k[1] for k in in_block})}")
    missing = [k for k in ta if k not in tb]
    if missing:
        print(f"tensors only in A: {len(missing)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true", help="also MD32, MDv32, MD64")
    ap.add_argument("--npz", type=Path, default=None, help="save the block tensors (not in B)")
    ap.add_argument("--sweep", action="store_true", help="fingerprints with ATEN_CPU_CAPABILITY default/avx2/avx512")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None)
    ap.add_argument("--npz-a", type=Path, default=None)
    ap.add_argument("--npz-b", type=Path, default=None)
    args = ap.parse_args()
    if args.compare:
        compare(args)
        return
    report(args)
    if args.sweep:
        sweep()


if __name__ == "__main__":
    main()
