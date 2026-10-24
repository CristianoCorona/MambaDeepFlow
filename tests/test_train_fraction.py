"""Tests of the data-quantity options (docs/lab-notebook.md, 2026-10-14): per-profile subset and step budget.

Only file names are used for the subset tests (no sample is loaded); test-set names come from a directory listing.
"""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mdf import data as mdata
from mdf import models

SEEDS = (0, 1, 2)
FRACTIONS = (0.25, 0.5)
FULL_STEPS = 133 * 599


@pytest.fixture(scope="module")
def split():
    train, val = mdata.split_files_grouped(mdata.list_files(mdata.TRAIN_DIR))
    return train, val


def test_every_profile_kept(split):
    train, _ = split
    profiles = {mdata.profile(f) for f in train}
    for seed in SEEDS:
        for f in FRACTIONS:
            sub = mdata.subsample_per_profile(train, f, seed)
            assert {mdata.profile(x) for x in sub} == profiles
            # exact count per profile: max(1, floor(f*n + 0.5))
            n = {}
            for x in train:
                n[mdata.profile(x)] = n.get(mdata.profile(x), 0) + 1
            k = {}
            for x in sub:
                k[mdata.profile(x)] = k.get(mdata.profile(x), 0) + 1
            assert all(k[p] == max(1, math.floor(f * n[p] + 0.5)) for p in profiles)


def test_nested_and_seed_dependent(split):
    train, _ = split
    for seed in SEEDS:
        s25 = set(mdata.subsample_per_profile(train, 0.25, seed))
        s50 = set(mdata.subsample_per_profile(train, 0.5, seed))
        assert s25 < s50
        assert mdata.subsample_per_profile(train, 0.5, seed) == mdata.subsample_per_profile(train, 0.5, seed)
    assert mdata.subsample_per_profile(train, 0.5, 0) != mdata.subsample_per_profile(train, 0.5, 1)


def test_no_validation_or_test_files(split):
    train, val = split
    test_names = set(mdata.list_files(mdata.TEST_DIR))
    for seed in SEEDS:
        for f in FRACTIONS:
            sub = set(mdata.subsample_per_profile(train, f, seed))
            assert sub <= set(train)
            assert not sub & set(val)
            assert not sub & test_names


def test_step_lr_matches_epoch_schedule():
    lr = 6e-4
    for step in range(FULL_STEPS):
        epoch = step // 599
        lr_e = models.compute_lr(epoch, 133, 0.1 * lr, lr)
        expected = lr_e if lr_e < lr else lr  # as the epoch loop of scripts/train.py
        assert models.step_lr(step, FULL_STEPS, 133, lr) == expected
    assert FULL_STEPS == 79667


def test_normal_split_unchanged(split):
    """Without train_fraction the training list is the one of the 133-epoch runs (sha recorded in night-1 configs)."""
    train, _ = split
    cfg = Path.home() / "runs" / "night1" / "ep133_s0" / "config.json"
    if not cfg.exists():
        pytest.skip("night-1 run not available")
    recorded = json.loads(cfg.read_text())["split"]["train_files_sha256"]
    assert mdata.file_list_sha256(train) == recorded


def test_print_counts(split):
    train, _ = split
    for f in FRACTIONS:
        counts = [len(mdata.subsample_per_profile(train, f, s)) for s in SEEDS]
        print(f"\nfraction {f}: samples per seed 0/1/2 = {counts} of {len(train)} "
              f"({100 * counts[0] / len(train):.1f} %), steps {FULL_STEPS}, "
              f"batches per data epoch {counts[0] // 10}, data epochs {FULL_STEPS / (counts[0] // 10):.1f}")
