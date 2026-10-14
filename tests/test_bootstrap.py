"""Tests of explore.paired_bootstrap_agg_median (aggregate / median ratio differences, notebook 2026-10-14)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mdf import explore


def _data(seed=0, n=30, seeds=3):
    rng = np.random.default_rng(seed)
    tgt = rng.uniform(1, 5, n)
    err = rng.uniform(0.1, 0.5, (seeds, n))
    profiles = [f"p{i // 3}" for i in range(n)]
    return err, tgt, err / tgt, profiles


def test_identical_configs_give_zero():
    err, tgt, rel, profiles = _data()
    r = explore.paired_bootstrap_agg_median(err, err, tgt, rel, rel, profiles, n_boot=200)
    assert r["agg_diff"] == 0 and r["median_diff"] == 0
    assert r["agg_ci1_lo"] == r["agg_ci1_hi"] == 0 and r["median_ci1_lo"] == r["median_ci1_hi"] == 0


def test_point_estimates_and_ci_contain_them():
    err, tgt, rel, profiles = _data()
    err2 = err * 1.2
    r = explore.paired_bootstrap_agg_median(err2, err, tgt, err2 / tgt, rel, profiles, n_boot=500)
    agg = err2.mean(0).sum() / tgt.sum() - err.mean(0).sum() / tgt.sum()
    med = np.median((err2 / tgt).mean(0)) - np.median(rel.mean(0))
    assert np.isclose(r["agg_diff"], agg) and np.isclose(r["median_diff"], med)
    assert r["agg_ci2_lo"] <= agg <= r["agg_ci2_hi"] and r["agg_ci1_lo"] > 0  # every resample: var is 20 % worse
