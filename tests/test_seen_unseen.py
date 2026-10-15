"""Tests of the difference-of-differences bootstrap of scripts/seen_unseen.py (synthetic data, no file read)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import seen_unseen as su


def synthetic(rng, shift_seen, shift_unseen):
    prof = {"seen": [f"s{i}" for i in range(40)], "unseen": [f"u{i // 4}" for i in range(48)]}
    b = {s: 0.02 + 0.005 * rng.random((3, len(prof[s]))) for s in su.SETS}
    v = {"seen": b["seen"] + shift_seen + 1e-4 * rng.standard_normal(b["seen"].shape),
         "unseen": b["unseen"] + shift_unseen + 1e-4 * rng.standard_normal(b["unseen"].shape)}
    return v, b, prof


def test_identical_configurations_give_zero():
    rng = np.random.default_rng(1)
    _, b, prof = synthetic(rng, 0.0, 0.0)
    r = su.dod_bootstrap(b, b, prof, 500, 0)
    for k in ("D_seen", "D_unseen", "DoD"):  # point estimates; the seeds of V and B are resampled independently
        assert abs(r[k]) < 1e-12
    assert r["DoD_lo"] < 0 < r["DoD_hi"]


def test_known_effect_recovered():
    rng = np.random.default_rng(2)
    v, b, prof = synthetic(rng, -0.002, 0.0)  # variant better only on seen shapes
    r = su.dod_bootstrap(v, b, prof, 2000, 0)
    assert abs(r["DoD"] + 0.002) < 1e-4
    assert r["DoD_lo"] <= r["DoD"] <= r["DoD_hi"] < 0
    assert r["D_unseen_lo"] < 0 < r["D_unseen_hi"]


def test_kendall_tau():
    assert su.kendall_tau([1, 2, 3], [10, 20, 30]) == 1
    assert su.kendall_tau([1, 2, 3], [30, 20, 10]) == -1


def test_reference_seeds_of_md8_unchanged_by_extra_b_seeds():
    """Night 6b: B has seeds 0-4; the DoD of MD8 / MDv8-16-32 must equal the one with B seeds 0-2 only."""
    rng = np.random.default_rng(5)
    profiles = {"seen": [f"p{i}" for i in range(40)], "unseen": [f"q{i // 3}" for i in range(60)]}
    v = {s: rng.normal(size=(3, len(profiles[s]))) for s in su.SETS}
    b5 = {s: rng.normal(size=(5, len(profiles[s]))) for s in su.SETS}
    for variant in ("MD8", "MDv8-16-32"):
        rows = su.ref_rows(variant, [0, 1, 2, 3, 4])
        assert rows == [0, 1, 2]
        r5 = su.dod_bootstrap(v, {s: b5[s][rows] for s in su.SETS}, profiles, 2000, 0)
        r3 = su.dod_bootstrap(v, {s: b5[s][:3] for s in su.SETS}, profiles, 2000, 0)
        assert r5 == r3
    assert su.ref_rows("DRv11", [0, 1, 2, 3, 4]) == [0, 1, 2, 3, 4]
    assert su.ref_rows("TR", [0, 2, 3]) == [0, 1, 2]
    assert su.ref_rows("MD8", [0, 1, 2]) == [0, 1, 2]  # night-6 setting: unchanged
