"""Tests of split mode grouped-full (full 53.8k set; docs/lab-notebook.md 2026-10-19, docs/preregistration-bigdata.md).

Synthetic file lists only (the full set is not on this machine), plus a tiny synthetic .npz directory for the loader.
Test-set names come from a directory listing; no test sample is read.

Run: conda run -n mdf python -m pytest -q tests/test_grouped_full.py
"""

import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mdf import data as mdata

# sha256 of the committed split files before this change (HEAD 4859a04, tag final-results-2026-10-18).
COMMITTED_SPLITS = {
    "kfold5_seed0_fold0.txt": "69ccf98c0d2662baa400b3a20f78ad5c9225dd02569189c8ac051a2497e67764",
    "kfold5_seed0_fold1.txt": "226d5f4be365377074a83dd158b48224de815dacd34f3b2e638828d8123aa976",
    "kfold5_seed0_fold2.txt": "0c38664d861106887c4ec27dddaad69e62e97b1a7cd3c040e2ea49b80cfdbd4c",
    "kfold5_seed0_fold3.txt": "b2506f58272bf58b0449a9382e162735cf3679127704b5881750db6882e10864",
    "kfold5_seed0_fold4.txt": "3b9f9157572b0162c9d62b35cc0f5f52cb7ce4fd08a8f6423fab087b9fab3fe5",
    "val_grouped_seed0.txt": "3b26466ac7ccc1ad6db6f9a0f2cbaffee01d11784e7d537d5e139e5af085e2e6",
    "val_seed0.txt": "7f6bc466c7937c2bf6e52c3e47afffa07e21c0c2e72cfe26fb05de9a75342023",
    "val_seen_holdout_seed0.txt": "b43a0fdbef9cd0d1f9950229fa519777f7bb171821a1ee9e9be62d4461882af2",
}
# sha256 of splits/test_profiles.txt (30 names, from the file names of data/test).
TEST_PROFILES_SHA256 = "9eddf8b3f14fda406906704e72255711850f82985f478368b7ade516ba7e1964"


def synthetic_full(seed=0, n_extra=200):
    """Full-set-like relative paths: every grouped validation profile and every test profile in reg/ and shear/,
    plus n_extra other profiles (one with a name ending in "_<int>", like the real "hybrid_1")."""
    rng = random.Random(seed)
    others = [f"af{i:04d}" for i in range(n_extra)] + ["hybrid_1", "davis_corrected"]
    profiles = mdata.grouped_val_profiles() + mdata.profiles_of_test() + others
    files = set()
    for p in profiles:
        for _ in range(rng.randint(1, 6)):
            files.add(f"reg/{p}_{rng.randint(500, 9999)}_{rng.randint(-2500, 2500)}.npz")
        for _ in range(rng.randint(0, 6)):
            files.add(f"shear/{p}_{rng.randint(-300, 300)}_{rng.randint(500, 9999)}_{rng.randint(-2500, 2500)}.npz")
    return sorted(files)


@pytest.fixture(scope="module")
def files():
    return synthetic_full()


@pytest.fixture(scope="module")
def parts(files):
    return mdata.split_files_grouped_full(files, mdata.grouped_val_profiles(), mdata.profiles_of_test())


def test_profile_rules():
    assert mdata.profile_full("reg/naca0012_5000_-300.npz") == "naca0012"
    assert mdata.profile_full("reg/hybrid_1_5000_300.npz") == "hybrid_1"  # regular airfoil named "hybrid_1"
    assert mdata.profile_full("shear/naca0012_-125_5000_300.npz") == "naca0012"
    assert mdata.profile_full("shear/hybrid_1_40_5000_300.npz") == "hybrid_1"
    assert mdata.profile_full("shear/davis_corrected_7_100_-5.npz") == "davis_corrected"
    for bad in ("reg/naca0012_x_1.npz", "shear/naca0012_5000_300.npz", "shear/a_b_1_2.npz", "sup/a_1_2.npz",
                "other/a_1_2.npz", "reg/a_1_2.npy", "reg/sub/a_1_2.npz"):
        with pytest.raises(ValueError):
            mdata.profile_full(bad)


def test_parts_disjoint_by_profile_and_cover(files, parts):
    train, val, excluded = parts
    assert sorted(train + val + excluded) == files
    assert all(p == sorted(p) for p in parts)
    prof = [{mdata.profile_full(f) for f in part} for part in parts]
    assert not prof[0] & prof[1] and not prof[0] & prof[2] and not prof[1] & prof[2]


def test_val_profiles_are_the_grouped_ones(parts):
    _, val, _ = parts
    grouped = {mdata.profile(f) for f in mdata.read_split_file(mdata.split_file_path("grouped"))}
    assert len(grouped) == 85
    assert {mdata.profile_full(f) for f in val} == grouped
    # every file of a validation profile is in val, from both subdirectories
    assert any(f.startswith("shear/") for f in val) and any(f.startswith("reg/") for f in val)


def test_no_test_profile_in_train_or_val(parts):
    train, val, excluded = parts
    test_p = set(mdata.profiles_of_test())
    assert len(test_p) == 30
    assert not {mdata.profile_full(f) for f in train + val} & test_p
    assert {mdata.profile_full(f) for f in excluded} == test_p


def test_overlapping_val_and_test_profiles_refused(files):
    with pytest.raises(ValueError):
        mdata.split_files_grouped_full(files, ["x", "y"], ["y"])


def test_test_profiles_file():
    text = mdata.TEST_PROFILES_FILE.read_bytes()
    assert hashlib.sha256(text).hexdigest() == TEST_PROFILES_SHA256
    names = mdata.profiles_of_test()
    assert names == sorted(set(names)) and len(names) == 30
    assert not set(names) & set(mdata.grouped_val_profiles())
    if mdata.TEST_DIR.exists():  # names only
        assert names == sorted({mdata.profile(f) for f in mdata.list_files(mdata.TEST_DIR)})


def test_existing_split_files_unchanged():
    for name, sha in COMMITTED_SPLITS.items():
        assert hashlib.sha256((mdata.SPLITS_DIR / name).read_bytes()).hexdigest() == sha, name
    assert set(mdata.ALL_SPLIT_MODES) == {"random", "grouped", "kfold", mdata.HOLDOUT}  # grouped-full not added


def test_existing_split_modes_unchanged():
    if not mdata.TRAIN_DIR.exists():
        pytest.skip("data/train not available")
    from test_holdout import EXISTING
    files = mdata.list_files(mdata.TRAIN_DIR)
    for (mode, fold), (sha_train, sha_val) in EXISTING.items():
        if mode == "kfold":
            train, val = mdata.split_files_for(mode, files, n_folds=5, fold=fold)
        else:
            train, val = mdata.split_files_for(mode, files)
        assert (mdata.file_list_sha256(train), mdata.file_list_sha256(val)) == (sha_train, sha_val), (mode, fold)


def test_full_subdirs():
    assert mdata.full_subdirs() == ("reg", "shear") or "MDF_FULL_SUBDIRS" in __import__("os").environ
    assert mdata.full_subdirs("reg") == ("reg",)
    for bad in ("", "reg,reg", "reg,foo", "test"):
        with pytest.raises(ValueError):
            mdata.full_subdirs(bad)


def test_list_files_full(tmp_path):
    for rel in ("reg/a_1_2.npz", "reg/b_1_2.npz", "shear/a_3_1_2.npz", "sup/c_1_2.npz", "reg/notes.txt"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"")
    assert mdata.list_files_full(tmp_path, ("reg", "shear")) == ["reg/a_1_2.npz", "reg/b_1_2.npz",
                                                                 "shear/a_3_1_2.npz"]
    assert mdata.list_files_full(tmp_path, ("reg",)) == ["reg/a_1_2.npz", "reg/b_1_2.npz"]
    assert mdata.list_files_full(tmp_path / "missing", ("reg",)) == []


def test_loader_on_synthetic_npz(tmp_path):
    """load_train_val_full / load_val_full on a few synthetic samples (random fields, valid normalisation)."""
    rng = np.random.default_rng(0)
    val_p, test_p = mdata.grouped_val_profiles()[:2], mdata.profiles_of_test()[:1]
    rels = [f"reg/{val_p[0]}_1000_10.npz", f"shear/{val_p[1]}_5_1000_10.npz", f"reg/{test_p[0]}_1000_10.npz"]
    rels += [f"reg/tr{i}_1000_{i}.npz" for i in range(5)] + [f"shear/tr{i}_7_1000_{i}.npz" for i in range(5)]
    for rel in rels:
        a = rng.standard_normal((6, 128, 128))
        a[0], a[1], a[2] = 10.0, 1.0, 0.0
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tmp_path / rel, a=a)
    train_ds, val_ds, info = mdata.load_train_val_full(full_dir=tmp_path, subdirs=("reg", "shear"))
    assert val_ds.files == sorted(rels[:2]) and len(train_ds) == 10
    assert info["n_excluded_test_profiles"] == 1 and info["n_files"] == 13
    assert info["n_files_per_subdir"] == {"reg": 7, "shear": 6}
    assert info["val_profiles_missing"] == sorted(set(mdata.grouped_val_profiles()) - set(val_p))
    assert info["val_files_sha256"] == mdata.file_list_sha256(val_ds.files)
    assert tuple(train_ds.inputs.shape) == (10, 3, 128, 128)
    assert mdata.load_val_full(full_dir=tmp_path, subdirs=("reg", "shear")).files == val_ds.files
    _, val_reg, info_reg = mdata.load_train_val_full(full_dir=tmp_path, subdirs=("reg",))
    assert val_reg.files == [rels[0]] and info_reg["n_train"] == 5
    sub, _, info_sub = mdata.load_train_val_full(full_dir=tmp_path, subdirs=("reg", "shear"), train_fraction=0.5,
                                                 subset_seed=0)
    assert len(sub) == 5 and info_sub["n_train_full"] == 10  # 2 files per profile -> k = 1
    with pytest.raises(FileNotFoundError):
        mdata.load_train_val_full(full_dir=tmp_path / "missing")
