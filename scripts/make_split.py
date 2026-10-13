"""Write the committed validation list for a split mode and check that profiles do not overlap.

Profile-disjoint split (docs/lab-notebook.md, D1 amended 2026-10-11): profile = name.rsplit("_", 2)[0];
with split seed 0, whole profiles are drawn at random until validation holds >= 400 files.
The random split (splits/val_seed0.txt) is only recomputed and checked, never rewritten.

K-fold (docs/lab-notebook.md, 2026-10-13): profile-disjoint folds of all files in data/train (train + val of
the other modes, never the test set), split seed 0; --write writes splits/kfold<N>_seed0_fold<k>.txt for every k.

Usage:
    python scripts/make_split.py [--split-mode grouped] [--write]
    python scripts/make_split.py --split-mode kfold --n-folds 5 [--write]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `mdf`

from mdf import data as mdata


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split-mode", choices=sorted(mdata.ALL_SPLIT_MODES), default="grouped")
    ap.add_argument("--n-folds", type=int, default=5, help="kfold only")
    ap.add_argument("--write", action="store_true", help="write the split file(s) (grouped / kfold only)")
    args = ap.parse_args()
    if args.split_mode == mdata.KFOLD:
        return kfold(args)

    files = mdata.list_files(mdata.TRAIN_DIR)
    train, val = mdata.SPLIT_MODES[args.split_mode](files, mdata.SPLIT_SEED)
    path = mdata.split_file_path(args.split_mode)
    p_train = {mdata.profile(f) for f in train}
    p_val = {mdata.profile(f) for f in val}
    shared = p_train & p_val
    print(f"split mode {args.split_mode}, split seed {mdata.SPLIT_SEED}, {len(files)} files, "
          f"{len(p_train | p_val)} profiles")
    print(f"train: {len(train)} files, {len(p_train)} profiles; val: {len(val)} files, {len(p_val)} profiles")
    print(f"files in both sets: {len(set(train) & set(val))}; train + val = all files: "
          f"{sorted(train + val) == files}")
    print(f"profiles shared by train and val: {len(shared)} {sorted(shared)[:10]}")
    print(f"val files sha256 {mdata.file_list_sha256(val)}; train files sha256 {mdata.file_list_sha256(train)}")

    if args.write:
        if args.split_mode != "grouped":
            sys.exit("Only the grouped split file is written by this script")
        path.write_text("".join(f + "\n" for f in val))
        print(f"wrote {path.relative_to(mdata.REPO_ROOT)}")
    if path.exists():
        print(f"{path.relative_to(mdata.REPO_ROOT)} matches the computed split: "
              f"{mdata.read_split_file(path) == val}")
    if args.split_mode == "grouped" and shared:
        sys.exit("Profiles overlap")


def kfold(args):
    files = mdata.list_files(mdata.TRAIN_DIR)
    n = args.n_folds
    folds = mdata.kfold_profile_folds(files, n, mdata.SPLIT_SEED)
    all_profiles = {mdata.profile(f) for f in files}
    print(f"kfold: {n} folds, split seed {mdata.SPLIT_SEED}, {len(files)} files, {len(all_profiles)} profiles "
          f"(directory data/train only)")
    seen = []
    ok = True
    for k in range(n):
        train, val = mdata.split_files_kfold(files, n, k, mdata.SPLIT_SEED)
        p_train = {mdata.profile(f) for f in train}
        p_val = {mdata.profile(f) for f in val}
        seen += val
        shared = p_train & p_val
        ok &= not shared and sorted(train + val) == files
        path = mdata.split_file_path(mdata.KFOLD, n_folds=n, fold=k)
        print(f"fold {k}: val {len(val)} files / {len(p_val)} profiles, train {len(train)} / {len(p_train)}; "
              f"shared profiles {len(shared)}; val sha256 {mdata.file_list_sha256(val)}")
        if args.write:
            path.write_text("".join(f + "\n" for f in val))
            print(f"  wrote {path.relative_to(mdata.REPO_ROOT)}")
        elif path.exists():
            print(f"  {path.relative_to(mdata.REPO_ROOT)} matches: {mdata.read_split_file(path) == val}")
    cover = sorted(seen) == files
    print(f"every file in exactly one validation fold: {cover}; folds pairwise disjoint in profiles: "
          f"{sum(len(f) for f in folds) == len(all_profiles) == len(set().union(*folds))}")
    if not (ok and cover):
        sys.exit("k-fold check failed")


if __name__ == "__main__":
    main()
