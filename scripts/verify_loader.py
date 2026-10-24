"""Verification 6a: the float32 loader (mdf/data.py) vs. the original loader (train/dataset.py).

Builds a temporary directory of symlinks to a sample of training files (fixed sample seed) and
to all test files, loads them with the original TurbDataset (float64) and with mdf.data
(float32), and reports per channel the max abs difference and the max abs difference relative
to the channel's max |value|. Comparison is against the original float64 arrays and against
the same arrays cast to float32 (what the original feeds to the network).

Also checks on ALL files that the mask (input channel 2) is binary, and that every target
channel is exactly 0 where the mask is 1 (raw and normalised).

Usage:
    python scripts/verify_loader.py [--n-train 200] [--sample-seed 0] [--tol 1e-6] [--skip-mask-check]
"""

import argparse
import contextlib
import io
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `mdf`

import numpy as np

from mdf import data as mdata
from mdf.models import ORIGINAL_DIR

NAMES = {"inputs": ["vx_in", "vy_in", "mask"], "targets": ["p", "vx", "vy"]}


def link_files(src_dir, files, dst_dir):
    dst_dir.mkdir(parents=True)
    for f in files:
        (dst_dir / f).symlink_to((src_dir / f).resolve())


def report(label, new, orig64, tol):
    worst = 0.0
    for kind in ("inputs", "targets"):
        for c, name in enumerate(NAMES[kind]):
            a = new[kind][:, c].astype(np.float64)
            b64 = orig64[kind][:, c]
            b32 = b64.astype(np.float32).astype(np.float64)
            scale = np.max(np.abs(b64)) or 1.0
            d64 = np.max(np.abs(a - b64))
            d32 = np.max(np.abs(a - b32))
            n_diff = int(np.count_nonzero(a != b32))
            worst = max(worst, d64)
            print(f"  {label:5s} {kind:7s} {name:6s} max|x| {scale:9.4f}  "
                  f"max|new-orig64| {d64:.3e} (rel {d64 / scale:.2e})  "
                  f"max|new-orig32| {d32:.3e}  elems!=orig32 {n_diff}")
    ok = worst <= tol
    print(f"  {label}: worst max|new-orig64| = {worst:.3e} -> {'PASS' if ok else 'FAIL'} (tol {tol:g})")
    return ok


def mask_check():
    bad = 0
    total = 0
    for directory in (mdata.TRAIN_DIR, mdata.TEST_DIR):
        for f in mdata.list_files(directory):
            with np.load(directory / f) as npf:
                a = npf["a"]
            m = a[2]
            total += 1
            if not np.all((m == 0) | (m == 1)) or np.any(a[3:6][:, m == 1] != 0):
                bad += 1
                print(f"  mask check failed: {directory.name}/{f}")
    print(f"Mask check (raw): {total} files, {bad} violations "
          "(mask in {0,1}; p, vx, vy == 0 inside the airfoil)")
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-6, help="max abs difference allowed vs. float64 original")
    ap.add_argument("--skip-mask-check", action="store_true")
    args = ap.parse_args()

    sys.path.append(str(ORIGINAL_DIR))
    import dataset as orig_dataset
    assert orig_dataset.__file__.startswith(str(ORIGINAL_DIR)), orig_dataset.__file__

    train_files = random.Random(args.sample_seed).sample(mdata.list_files(mdata.TRAIN_DIR), args.n_train)
    test_files = mdata.list_files(mdata.TEST_DIR)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        link_files(mdata.TRAIN_DIR, train_files, tmp / "train")
        link_files(mdata.TEST_DIR, test_files, tmp / "test")

        # original, train mode, no shuffle: sorted files; undo its 80/20 split
        with contextlib.redirect_stdout(io.StringIO()):
            o = orig_dataset.TurbDataset(None, mode=orig_dataset.TurbDataset.TRAIN,
                                         dataDir=str(tmp / "train") + "/", shuffle=0)
        orig_train = {"inputs": np.concatenate([o.inputs, o.valiInputs]),
                      "targets": np.concatenate([o.targets, o.valiTargets])}
        assert orig_train["inputs"].dtype == np.float64
        new_train = mdata.FlowDataset(tmp / "train", sorted(train_files))
        new_train = {"inputs": new_train.inputs.numpy(), "targets": new_train.targets.numpy()}
        assert new_train["inputs"].dtype == np.float32

        # original, test mode (loads 10 train files first, then replaces them with the test files)
        with contextlib.redirect_stdout(io.StringIO()):
            ot = orig_dataset.TurbDataset(None, mode=orig_dataset.TurbDataset.TEST,
                                          dataDir=str(tmp / "train") + "/", dataDirTest=str(tmp / "test") + "/")
        orig_test = {"inputs": ot.inputs, "targets": ot.targets}
        nt = mdata.FlowDataset(tmp / "test", test_files)
        new_test = {"inputs": nt.inputs.numpy(), "targets": nt.targets.numpy()}

        print(f"Samples compared: train {len(train_files)} (sample seed {args.sample_seed}), test {len(test_files)}")
        ok &= report("train", new_train, orig_train, args.tol)
        ok &= report("test", new_test, orig_test, args.tol)

        # denormalisation: mdf vs. the original TurbDataset.denormalize on the same sample
        x = new_test["targets"][0]
        v = mdata.v_norm_from_normalized_input(new_test["inputs"][0])
        dd = np.max(np.abs(mdata.denormalize(x, v) - ot.denormalize(x, v)))
        print(f"Denormalize (test sample 0): max|mdf - original| = {dd:.3e}")
        ok &= dd == 0.0

        # normalised targets are still exactly 0 inside the airfoil
        for label, d in (("train", new_train), ("test", new_test)):
            inside = np.broadcast_to(d["inputs"][:, 2:3] == 1, d["targets"].shape)
            print(f"Normalised targets inside airfoil ({label}): max |t| = {np.max(np.abs(d['targets'][inside])):.1e}")

    if not args.skip_mask_check:
        ok &= mask_check()
    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
