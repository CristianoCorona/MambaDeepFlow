"""Write the committed validation list for a split mode and check that profiles do not overlap.

Profile-disjoint split (docs/lab-notebook.md, D1 amended 2026-10-11): profile = name.rsplit("_", 2)[0];
with split seed 0, whole profiles are drawn at random until validation holds >= 400 files.
The random split (splits/val_seed0.txt) is only recomputed and checked, never rewritten.

K-fold (docs/lab-notebook.md, 2026-10-13): profile-disjoint folds of all files in data/train (train + val of
the other modes, never the test set), split seed 0; --write writes splits/kfold<N>_seed0_fold<k>.txt for every k.

Grouped-holdout (docs/lab-notebook.md, 2026-10-15): unseen-shape validation = the grouped one (checked against
splits/val_grouped_seed0.txt), seen-shape validation = 400 files of the grouped training part, one per profile with
>= 2 files; --write writes splits/val_seen_holdout_seed0.txt only.

Usage:
    python scripts/make_split.py [--split-mode grouped] [--write]
    python scripts/make_split.py --split-mode kfold --n-folds 5 [--write]
    python scripts/make_split.py --split-mode grouped-holdout [--write]
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
    if args.split_mode == mdata.HOLDOUT:
        return holdout(args)

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


def holdout(args):
    files = mdata.list_files(mdata.TRAIN_DIR)
    train, unseen, seen = mdata.split_files_grouped_holdout_parts(files, mdata.SPLIT_SEED)
    g_train, g_val = mdata.split_files_grouped(files, mdata.SPLIT_SEED)
    p_train = {mdata.profile(f) for f in train}
    p_unseen = {mdata.profile(f) for f in unseen}
    p_seen = [mdata.profile(f) for f in seen]
    test_names = set(mdata.list_files(mdata.TEST_DIR))  # names only, nothing is read
    checks = {
        "parts pairwise disjoint": not (set(train) & set(unseen) or set(train) & set(seen) or set(unseen) & set(seen)),
        "train + unseen + seen = all files": sorted(train + unseen + seen) == files,
        "unseen = grouped validation": unseen == g_val,
        "unseen matches splits/val_grouped_seed0.txt": mdata.read_split_file(mdata.split_file_path("grouped")) == unseen,
        "train = grouped train minus seen": train == [f for f in g_train if f not in set(seen)],
        "seen: one file per profile": len(set(p_seen)) == len(seen) == mdata.N_SEEN,
        "every seen profile has >= 1 training file": set(p_seen) <= p_train,
        "no unseen profile in train": not p_unseen & p_train,
        "no test file name in any part": not test_names & set(files),
    }
    print(f"grouped-holdout, split seed {mdata.SPLIT_SEED}: train {len(train)} files / {len(p_train)} profiles; "
          f"unseen {len(unseen)} / {len(p_unseen)}; seen {len(seen)} / {len(set(p_seen))}")
    print(f"sha256 train {mdata.file_list_sha256(train)}; unseen {mdata.file_list_sha256(unseen)}; "
          f"seen {mdata.file_list_sha256(seen)}")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    path = mdata.seen_split_file_path()
    if args.write:
        path.write_text("".join(f + "\n" for f in seen))
        print(f"wrote {path.relative_to(mdata.REPO_ROOT)}")
    if path.exists():
        print(f"{path.relative_to(mdata.REPO_ROOT)} matches the computed split: {mdata.read_split_file(path) == seen}")
    if not all(checks.values()):
        sys.exit("grouped-holdout check failed")


if __name__ == "__main__":
    main()
