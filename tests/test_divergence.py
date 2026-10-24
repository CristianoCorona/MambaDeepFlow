"""Unit tests of the divergence helpers in mdf/explore.py (docs/lab-notebook.md, 2026-10-14)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mdf import explore

N = 16
I, J = np.mgrid[0:N, 0:N].astype(np.float64)  # I: array axis 0 (rows), J: axis 1 (cols)
INNER = (slice(1, -1), slice(1, -1))


@pytest.mark.parametrize("ax", [0, 1])
def test_analytic_fields(ax):
    x, y = (I, J) if ax == 0 else (J, I)  # coordinate along which vx is differentiated, and the other one
    mp = (ax, 1, 1)
    div0 = explore.divergence(x, -y, mp, scale=(1, 1))  # (x, -y): divergence 0
    div2 = explore.divergence(x, y, mp, scale=(1, 1))   # (x, y): divergence 2
    assert np.allclose(div0[INNER], 0.0)
    assert np.allclose(div2[INNER], 2.0)


def test_sign_flip_and_scale():
    vx, vy = np.sin(I / 3) * J, np.cos(J / 4) + I ** 2 / 50
    for ax in (0, 1):
        a = explore.divergence(vx, vy, (ax, 1, -1), scale=(1, 1))
        b = explore.divergence(vx, vy, (ax, -1, 1), scale=(1, 1))
        assert np.allclose(a, -b)  # flipping both signs only flips the sign
    s = explore.divergence(vx, vy, (0, 1, 1), scale=(2.0, 3.0))
    assert np.allclose(s, explore.divergence(2 * vx, 3 * vy, (0, 1, 1), scale=(1, 1)))
    # default scale = the fixed maxima of vx and vy (2.04, 2.37)
    assert np.allclose(explore.divergence(vx, 0 * vy, (0, 1, 1)), 2.04 * explore.divergence(vx, 0 * vy, (0, 1, 1),
                                                                                             scale=(1, 1)))


def test_valid_stencil():
    fluid = np.ones((6, 6), dtype=bool)
    fluid[3, 3] = False  # one solid cell
    v = explore.valid_stencil(fluid)
    assert not v[0].any() and not v[-1].any() and not v[:, 0].any() and not v[:, -1].any()  # grid edge
    assert not v[3, 3] and not v[2, 3] and not v[4, 3] and not v[3, 2] and not v[3, 4]  # solid and its neighbours
    assert v[2, 2] and v[1, 1]  # diagonal neighbour is fine (4-neighbour stencil)
    assert v.sum() == 16 - 5
