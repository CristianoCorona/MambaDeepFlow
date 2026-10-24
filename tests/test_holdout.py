"""Tests of the grouped-holdout split (docs/lab-notebook.md, 2026-10-15, seen-shape vs unseen-shape validation).

File lists only, except the seen-shape loader test (400 samples). Test-set names come from a directory listing; no
test sample is read.

Run: conda run -n mdf python -m pytest -q tests/test_holdout.py
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mdf import data as mdata
from mdf import explore

# sha256 of (train, val) file lists of the existing split modes, computed with the code before this change (HEAD
# d46b279, 2026-10-15): they must not change.
EXISTING = {
    ("random", None): ("636f439591a8d6164ba224094354474261b5d8d0d9e10ec2988ac62b026923cf",
                       "7f6bc466c7937c2bf6e52c3e47afffa07e21c0c2e72cfe26fb05de9a75342023"),
    ("grouped", None): ("351af04002f93ec247b049967c82f3053fabe1881eb89855909d437fff1e5daf",
                        "3b26466ac7ccc1ad6db6f9a0f2cbaffee01d11784e7d537d5e139e5af085e2e6"),
    ("kfold", 0): ("afedee449e3243a131fa794c3a27466a50ae27f2452fd1a3238599de17fa6ea8",
                   "69ccf98c0d2662baa400b3a20f78ad5c9225dd02569189c8ac051a2497e67764"),
    ("kfold", 1): ("bd92f10a09803bbbd17e706af5e478d376145748383edb873ee82246dc906b90",
                   "226d5f4be365377074a83dd158b48224de815dacd34f3b2e638828d8123aa976"),
    ("kfold", 2): ("4407f8ee57a927ea53be4818f6d08b074ff044838b6a2d9cddd941dcbc4bee1c",
                   "0c38664d861106887c4ec27dddaad69e62e97b1a7cd3c040e2ea49b80cfdbd4c"),
    ("kfold", 3): ("630b1cedcd54e8d973f76188d4985fc2e13f65249291515fa636bbae78e503e6",
                   "b2506f58272bf58b0449a9382e162735cf3679127704b5881750db6882e10864"),
    ("kfold", 4): ("5196f87f9461cbd6650bf24e9da6fe829a5ec11e2424f62c07de33b90ea3e749",
                   "3b9f9157572b0162c9d62b35cc0f5f52cb7ce4fd08a8f6423fab087b9fab3fe5"),
}


@pytest.fixture(scope="module")
def files():
    return mdata.list_files(mdata.TRAIN_DIR)


@pytest.fixture(scope="module")
def parts(files):
    return mdata.split_files_grouped_holdout_parts(files)


def test_parts_disjoint_and_cover(files, parts):
    train, unseen, seen = parts
    assert not set(train) & set(unseen)
    assert not set(train) & set(seen)
    assert not set(unseen) & set(seen)
    assert sorted(train + unseen + seen) == files
    assert all(p == sorted(p) for p in parts)
    assert (len(train), len(unseen), len(seen)) == (5598, 402, mdata.N_SEEN)


def test_seen_profiles_in_training(parts):
    train, _, seen = parts
    p_train = {}
    for f in train:
        p_train[mdata.profile(f)] = p_train.get(mdata.profile(f), 0) + 1
    p_seen = [mdata.profile(f) for f in seen]
    assert len(set(p_seen)) == len(seen)  # at most one file per profile
    assert all(p_train.get(p, 0) >= 1 for p in p_seen)  # every seen shape stays in training


def test_unseen_is_grouped_validation(files, parts):
    g_train, g_val = mdata.split_files_grouped(files)
    train, unseen, seen = parts
    assert unseen == g_val
    assert unseen == mdata.read_split_file(mdata.split_file_path("grouped"))
    assert mdata.split_file_path(mdata.HOLDOUT) == mdata.split_file_path("grouped")
    assert train == [f for f in g_train if f not in set(seen)]
    assert not {mdata.profile(f) for f in unseen} & {mdata.profile(f) for f in train}
    assert mdata.split_files_for(mdata.HOLDOUT, files) == (train, unseen)


def test_committed_seen_list(parts):
    assert mdata.read_split_file(mdata.seen_split_file_path()) == parts[2]


def test_no_test_files(parts):
    test_names = set(mdata.list_files(mdata.TEST_DIR))
    assert test_names, "test directory listing is empty"
    for part in parts:
        assert not set(part) & test_names


def test_existing_splits_unchanged(files):
    assert set(mdata.ALL_SPLIT_MODES) == {"random", "grouped", "kfold", mdata.HOLDOUT}
    for (mode, fold), (sha_train, sha_val) in EXISTING.items():
        n_folds = 5 if mode == "kfold" else None
        train, val = mdata.split_files_for(mode, files, n_folds=n_folds, fold=fold)
        assert mdata.file_list_sha256(train) == sha_train, (mode, fold)
        assert mdata.file_list_sha256(val) == sha_val, (mode, fold)
        assert mdata.read_split_file(mdata.split_file_path(mode, n_folds=n_folds, fold=fold)) == val


def test_existing_run_split_unchanged(files):
    """The grouped split of the 133-epoch runs is the one recorded in their config.json."""
    cfg = Path.home() / "runs" / "night1" / "ep133_s0" / "config.json"
    if not cfg.exists():
        pytest.skip("night-1 run not available")
    rec = json.loads(cfg.read_text())["split"]
    train, val = mdata.split_files_grouped(files)
    assert (mdata.file_list_sha256(train), mdata.file_list_sha256(val)) == (rec["train_files_sha256"],
                                                                            rec["val_files_sha256"])


def test_seen_loaders(parts):
    ds = mdata.load_val_seen(split_file=mdata.seen_split_file_path())
    assert ds.files == parts[2] and len(ds) == mdata.N_SEEN
    ds2 = explore.load_split("val_seen", mdata.HOLDOUT)
    assert ds2.files == parts[2]
    with pytest.raises(ValueError):
        explore.load_split("val_seen", "grouped")
