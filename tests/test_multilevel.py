"""Unit tests for multi-level models (one block per level, docs/lab-notebook.md 2026-10-15).

Run: python -m pytest -s tests/test_multilevel.py   (needs CUDA for the mamba_ssm kernels)

Shapes for every combination of >= 2 levels, identity with B at initialisation (zero out_proj in every block),
non-zero gradients on every block parameter after two Adam steps, parameter counts (printed), configuration
handling, and a fingerprint check that the single-level variants are unchanged by the multi-level code.
"""

import hashlib
import itertools
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mdf import models  # noqa: E402
from test_blocks import build, inputs, needs_cuda  # noqa: E402

COMBOS = [c for k in range(2, len(models.LEVELS) + 1) for c in itertools.combinations(sorted(models.LEVELS), k)]


def build_multi(variant, levels, seed=0):
    torch.manual_seed(seed)
    return models.build_from_cfg({"variant": variant, "levels": levels})


def fingerprint(net):
    h = hashlib.sha256()
    for k, v in net.state_dict().items():
        h.update(k.encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


# Initial state dicts (seed 0, CPU) computed with the code of commit 686398f, BEFORE the multi-level change
# (git worktree of that commit; same values after the change).
FINGERPRINTS = {
    ("B", None): "fce1134b74f28063",
    ("B-large", None): "fea7dd870678a760",
    ("CBAM", 8): "a0cd78bff043b447",
    ("SA", 8): "393780ffbad8a9aa",
    ("MD", 8): "8af23666a0e08dad",
    ("MDv", 8): "757cec5dc1e9167e",
    ("SA", 32): "d467c0387cd646b3",
    ("MD", 32): "3b708347e0852f60",
    ("MDv", 32): "c6b2eb19bf154401",
    ("MDv", 64): "f8ec7c0afa06678a",
    ("MD", 64): "b4c347cf5ae590b7",
}


# 1 -------------------------------------------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("variant", ["MDv", "MD"])
@pytest.mark.parametrize("levels", COMBOS)
def test_shapes(variant, levels):
    net = build_multi(variant, levels).cuda()
    out = net(inputs(2, device="cuda"))
    assert out.shape == (2, 3, 128, 128) and torch.isfinite(out).all()
    assert sorted(int(k) for k in net.mixers) == sorted(levels)


# 2 -------------------------------------------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("variant", ["MDv", "MD", "SA"])
@pytest.mark.parametrize("levels", [(8, 16, 32), (32, 64), (8, 16, 32, 64)])
def test_zero_init_equals_B(variant, levels):
    b = build("B").cuda().eval()
    net = build_multi(variant, levels).cuda().eval()
    skeleton = {k: v for k, v in net.state_dict().items() if not k.startswith("mixers.")}
    ref = b.state_dict()
    assert skeleton.keys() == ref.keys() and all(torch.equal(skeleton[k], ref[k]) for k in ref)
    x = inputs(2, device="cuda")
    with torch.no_grad():
        diff = (net(x) - b(x)).abs().max().item()
    print(f"{variant}{models.levels_tag(levels)}: max |out - B| at init {diff:.3g}")
    assert diff < 1e-5


# 3 -------------------------------------------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("variant", ["MDv", "MD"])
def test_gradients_after_two_adam_steps(variant):
    net = build_multi(variant, (8, 16, 32)).cuda().train()
    opt = torch.optim.Adam(net.parameters(), lr=6e-4, betas=(0.5, 0.999))
    x, t = inputs(4, device="cuda"), inputs(4, seed=7, device="cuda")
    for _ in range(2):
        opt.zero_grad()
        torch.nn.functional.l1_loss(net(x), t).backward()
        opt.step()
    zero = [n for n, p in net.mixers.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not zero, f"parameters without gradient after two steps: {zero}"


# 4 -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["MDv", "MD"])
def test_param_counts(variant):
    single = {lv: models.block_params(build(variant, lv)) for lv in models.LEVELS}
    for levels in [(8, 16, 32), (32, 64), (8, 64)]:
        net = build_multi(variant, levels)
        n_block = models.block_params(net)
        print(f"{variant}{models.levels_tag(levels)}: block params {n_block} "
              f"({' + '.join(f'{lv}:{single[lv]}' for lv in sorted(levels))}), total {models.count_params(net)}")
        assert n_block == sum(single[lv] for lv in levels)
        assert models.count_params(net) == models.B_PARAMS + n_block


# 5 -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", list(FINGERPRINTS))
def test_existing_variants_unchanged(key):
    variant, level = key
    net = build(variant, level)
    assert not hasattr(net, "mixers")
    assert fingerprint(net) == FINGERPRINTS[key]


# 6 -------------------------------------------------------------------------------------------------------------
def test_config_handling():
    cfg = models.normalize_cfg({"variant": "MDv", "levels": "8,16,32"})
    assert cfg == {"variant": "MDv", "level": None, "levels": (32, 16, 8), "channels": 32, "mamba_impl": "cuda"}
    assert models.levels_tag(cfg["levels"]) == "8-16-32"
    assert "levels" not in models.normalize_cfg({"variant": "MDv", "level": 8})
    for bad in ({"variant": "B", "levels": "8,16"}, {"variant": "MDv", "levels": "8"},
                {"variant": "MDv", "levels": "8,12"}, {"variant": "MDv", "level": 8, "levels": "16,32"}):
        with pytest.raises(ValueError):
            models.normalize_cfg(bad)
    # run config.json round trip (train.py writes levels as a list; single-level runs have levels None or no key)
    run_cfg = {"variant": "MDv", "level": None, "levels": [32, 16, 8], "channels": 32, "mamba_impl": "cuda"}
    assert models.model_cfg_from_run_config(run_cfg) == cfg
    assert models.model_cfg_from_run_config({"variant": "MD", "level": 8, "levels": None, "channels": 32}) == \
        models.normalize_cfg({"variant": "MD", "level": 8})


def test_checkpoint_round_trip():
    net = build_multi("SA", (8, 16, 32), seed=0)
    for p in net.mixers.parameters():
        torch.nn.init.normal_(p)
    other = build_multi("SA", (8, 16, 32), seed=1)
    other.load_state_dict(net.state_dict(), strict=True)
    x = inputs(1)
    with torch.no_grad():
        assert torch.equal(net.eval()(x), other.eval()(x))
