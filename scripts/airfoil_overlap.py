"""Read-only check: airfoil profiles shared between the training, validation and test files.

data/dataGen.py names each sample "<airfoil>_<int(fsX*100)>_<int(fsY*100)>.npz", where <airfoil>
is the basename of the airfoil coordinate file (which may itself contain underscores, e.g.
"davis_corrected"). The profile is therefore name.rsplit("_", 2)[0]. Only file names are read;
the split is the fixed one in splits/val_seed0.txt and is not changed.

Limitation: identical names imply the same coordinate file; different names may still be
near-identical shapes (e.g. variants of one family), which this check cannot detect.

Usage:
    python scripts/airfoil_overlap.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `mdf`

from mdf import data as mdata


def profile(name):
    stem = name[:-len(".npz")]
    airfoil, fsx, fsy = stem.rsplit("_", 2)
    int(fsx), int(fsy)  # both must be integers, else the naming assumption is wrong
    return airfoil


def main():
    all_train = mdata.list_files(mdata.TRAIN_DIR)
    test = mdata.list_files(mdata.TEST_DIR)
    train, val = mdata.split_files(all_train)
    if sorted(val) != sorted(mdata.read_split_file(mdata.SPLITS_DIR / "val_seed0.txt")):
        sys.exit("Recomputed split differs from splits/val_seed0.txt")

    sets = {"train+val (6.4k)": all_train, "train": train, "val": val, "test": test}
    prof = {k: Counter(profile(f) for f in v) for k, v in sets.items()}
    print(f"{'set':18s} {'files':>6s} {'profiles':>9s} {'files/profile (min/median/max)':>32s}")
    for k, c in prof.items():
        counts = sorted(c.values())
        print(f"{k:18s} {sum(counts):6d} {len(c):9d} "
              f"{counts[0]:>14d} / {counts[len(counts) // 2]} / {counts[-1]}")

    tr, va, te = set(prof["train"]), set(prof["val"]), set(prof["test"])
    val_shared_files = sum(1 for f in val if profile(f) in tr)
    print(f"\nval profiles also in train: {len(va & tr)} / {len(va)}; "
          f"val files whose profile is in train: {val_shared_files} / {len(val)}")
    print(f"val profiles not in train: {sorted(va - tr)}")
    all_tr = set(prof["train+val (6.4k)"])
    print(f"test profiles also in train+val: {len(te & all_tr)} / {len(te)}"
          f" (in train: {len(te & tr)}, in val: {len(te & va)})")
    if te & all_tr:
        print(f"shared test profiles: {sorted(te & all_tr)}")
    print(f"test files: {len(test)}, test files with profile in train+val: "
          f"{sum(1 for f in test if profile(f) in all_tr)}")


if __name__ == "__main__":
    main()
