"""Run metadata: seeding, git state, library/GPU versions, timers, peak memory."""

import json
import platform
import random
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
import torch

from mdf.data import REPO_ROOT


def seed_everything(seed, deterministic=False):
    """Seed random, numpy, torch (CPU and all CUDA devices). Returns a seeded DataLoader generator."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info():
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "dirty_files": None if not status else status.splitlines(),
    }


def env_info():
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_capability"] = f"{p.major}.{p.minor}"
        info["gpu_total_mib"] = round(p.total_memory / 2**20)
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True)
        info["nvidia_driver"] = smi.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        info["nvidia_driver"] = None
    return info


def peak_rss_mib():
    """Peak resident set size of this process (ru_maxrss is in KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def peak_vram():
    if not torch.cuda.is_available():
        return {}
    return {
        "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timers:
    """Accumulating wall-clock timers; CUDA is synchronised at every boundary."""

    def __init__(self):
        self.totals = {}

    @contextmanager
    def __call__(self, name):
        cuda_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            cuda_sync()
            self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - t0


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def command_line():
    return [sys.executable.split("/")[-1], *sys.argv]


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
        fh.write("\n")
