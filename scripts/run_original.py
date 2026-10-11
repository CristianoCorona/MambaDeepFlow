"""Run the UNMODIFIED train/runTrain.py with an explicit seed and all outputs in a run directory.

runTrain.py draws its own seed with random.randint(0, 2**32 - 1); seeding Python's `random`
with --seed right before executing it makes that draw (and therefore the whole run setup)
reproducible. The derived seed is recorded in config.json (deviation D9 in docs/lab-notebook.md).

runTrain.py uses relative paths ("../data/train/", "results_train/") and writes `<prefix>modelG`.
To keep train/ clean, the script is executed with cwd = <run-dir>/work, where <run-dir>/data is a
symlink to the repo's data/ directory, and with prefix = "<run-dir>/". The original modules
(DfpNet, dataset, utils) are imported from train/, appended at the END of sys.path.

Outputs: <run-dir>/modelG (+ model.pt symlink for scripts/evaluate.py), <run-dir>/work/results_train/,
<run-dir>/config.json. Redirect stdout to keep the original log (e.g. > <run-dir>/train.log).

Usage:
    python scripts/run_original.py --seed 0 [--out-dir DIR]
"""

import argparse
import os
import random
import runpy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `mdf`

import torch

from mdf import runinfo
from mdf.data import REPO_ROOT
from mdf.models import ORIGINAL_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    run_dir = (args.out_dir or Path.home() / "runs" / f"orig_dfpnet_s{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        sys.exit(f"Run directory {run_dir} exists and is not empty")
    (run_dir / "work").mkdir(parents=True, exist_ok=True)
    (run_dir / "data").symlink_to(REPO_ROOT / "data", target_is_directory=True)

    derived_seed = random.Random(args.seed).randint(0, 2**32 - 1)
    config = {
        "run_id": run_dir.name,
        "status": "running",
        "started_utc": runinfo.utc_now(),
        "command": runinfo.command_line(),
        "script": "train/runTrain.py (unmodified)",
        "model": "dfpnet",
        "args": {"expo": 5, "seed": args.seed},
        "seed": args.seed,
        "derived_seed_expected": derived_seed,
        "note": "original per-seed train/val split (shuffle=1), float64 data, 10k requested iterations",
        "git": runinfo.git_info(),
        "env": runinfo.env_info(),
    }
    runinfo.write_json(run_dir / "config.json", config)

    sys.path.append(str(ORIGINAL_DIR))
    # import the original modules (and matplotlib/PIL) before seeding, so no import can consume `random`
    import DfpNet, dataset, utils  # noqa: E401,F401
    sys.argv = [str(ORIGINAL_DIR / "runTrain.py"), str(run_dir) + "/"]
    cwd = Path.cwd()
    os.chdir(run_dir / "work")
    t0 = time.perf_counter()
    try:
        random.seed(args.seed)
        runpy.run_path(str(ORIGINAL_DIR / "runTrain.py"), run_name="__main__")
    finally:
        os.chdir(cwd)
    total = time.perf_counter() - t0

    loaded = {name: getattr(sys.modules.get(name), "__file__", None) for name in ("DfpNet", "dataset", "utils")}
    for name, path in loaded.items():
        if not path or not path.startswith(str(ORIGINAL_DIR)):
            sys.exit(f"{name} was imported from {path}, not from {ORIGINAL_DIR}")
    (run_dir / "model.pt").symlink_to("modelG")

    config.update({
        "status": "finished",
        "finished_utc": runinfo.utc_now(),
        "original_modules": {k: str(Path(v).relative_to(REPO_ROOT)) for k, v in loaded.items()},
        "timing_seconds": {"runTrain_total": total},
        "peak_rss_mib": runinfo.peak_rss_mib(),
        "peak_vram": runinfo.peak_vram(),
    })
    runinfo.write_json(run_dir / "config.json", config)
    print(f"run_original: derived seed {derived_seed}, {total:.1f}s, peak RSS {config['peak_rss_mib']:.0f} MiB, "
          f"peak VRAM alloc {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
