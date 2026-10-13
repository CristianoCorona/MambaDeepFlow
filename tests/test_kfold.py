"""Tests of the profile-disjoint k-fold split (docs/lab-notebook.md, 2026-10-13). File lists only, no data loaded.

Run: conda run -n mdf python -m pytest -q tests/test_kfold.py
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mdf import data as mdata

N_FOLDS = 5


@pytest.fixture(scope="module")
def files():
    return mdata.list_files(mdata.TRAIN_DIR)


@pytest.fixture(scope="module")
def splits(files):
    return [mdata.split_files_kfold(files, N_FOLDS, k) for k in range(N_FOLDS)]


def test_folds_cover_every_profile_exactly_once(files):
    folds = mdata.kfold_profile_folds(files, N_FOLDS)
    profiles = {mdata.profile(f) for f in files}
    assert len(profiles) == 1456
    assert set().union(*folds) == profiles
    assert sum(len(f) for f in folds) == len(profiles)  # no profile in two folds


def test_each_file_in_exactly_one_validation_fold(files, splits):
    vals = [f for _, val in splits for f in val]
    assert sorted(vals) == files and len(vals) == len(set(vals)) == 6400


def test_train_val_disjoint_and_complete(files, splits):
    for train, val in splits:
        assert not set(train) & set(val)
        assert sorted(train + val) == files
        assert not {mdata.profile(f) for f in train} & {mdata.profile(f) for f in val}


def test_folds_balanced(splits):
    sizes = [len(val) for _, val in splits]
    assert max(sizes) - min(sizes) <= 20, sizes  # greedy balancing by file count


def test_no_test_files(files, splits):
    train_dir, test_dir = mdata.TRAIN_DIR.resolve(), mdata.TEST_DIR.resolve()
    assert train_dir != test_dir
    for _, val in splits:
        for f in val:
            path = (mdata.TRAIN_DIR / f).resolve()
            assert path.parent == train_dir and test_dir not in path.parents
    # names only (os.listdir), no test file is opened
    test_names = set(os.listdir(test_dir)) if test_dir.exists() else set()
    assert len(test_names) in (0, 90)
    assert not test_names & set(files)


def test_deterministic_and_committed_files(files, splits):
    again = [mdata.split_files_kfold(files, N_FOLDS, k) for k in range(N_FOLDS)]
    assert again == splits
    for k, (_, val) in enumerate(splits):
        path = mdata.split_file_path(mdata.KFOLD, n_folds=N_FOLDS, fold=k)
        assert path.name == f"kfold{N_FOLDS}_seed0_fold{k}.txt"
        assert mdata.read_split_file(path) == val


def test_existing_split_modes_unchanged(files):
    _, val = mdata.split_files_for("grouped", files)
    assert mdata.file_list_sha256(val) == "3b26466ac7ccc1ad6db6f9a0f2cbaffee01d11784e7d537d5e139e5af085e2e6"
    assert mdata.read_split_file(mdata.split_file_path("grouped")) == val
    _, val_r = mdata.split_files_for("random", files)
    assert mdata.read_split_file(mdata.split_file_path("random")) == val_r


def test_bad_arguments(files):
    with pytest.raises(ValueError):
        mdata.split_files_kfold(files, N_FOLDS, N_FOLDS)
    with pytest.raises(ValueError):
        mdata.split_files_for(mdata.KFOLD, files)
    with pytest.raises(ValueError):
        mdata.kfold_profile_folds(files, 1)
