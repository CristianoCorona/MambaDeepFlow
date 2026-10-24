"""Unit tests for Conv64 (night 8b, mdf/ablation.ConvResBlock64; docs/lab-notebook.md 2026-10-17): a plain residual
conv block y = x + Conv3x3(12 -> 32)(LeakyReLU(0.2)(Conv3x3(32 -> 12)(x))) after layer1 (32 channels, 64x64).

Run: python -m pytest -s tests/test_conv64.py

Checks: shapes, parameter counts, skeleton weights equal to B at the same seed, output identical to B at
initialisation (zero last convolution), non-zero gradient on every block parameter after two Adam steps, locality
(a one-pixel perturbation changes the block output exactly in the 5x5 neighbourhood and nowhere else), config handling.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mdf import ablation, blocks, models  # noqa: E402
from test_blocks import DEV, inputs, original_turbnet  # noqa: E402

CFG = {"variant": "Conv", "level": 64}
C, H = 32, 12
BLOCK_PARAMS = (C * H * 9 + H) + (H * C * 9 + C)  # conv1 + conv2, with bias = 6956


def build_conv(seed=0):
    torch.manual_seed(seed)
    return models.build_from_cfg(CFG)


def test_shapes():
    net = build_conv().to(DEV).train()
    assert isinstance(net.mixer, ablation.ConvResBlock64) and net.mix_after == "layer1"
    seen = []
    h = net.mixer.register_forward_hook(lambda m, i, o: seen.append((tuple(i[0].shape), tuple(o.shape))))
    y = net(inputs(device=DEV))
    h.remove()
    assert y.shape == (2, 3, 128, 128) and torch.isfinite(y).all()
    assert seen == [((2, C, 64, 64), (2, C, 64, 64))], seen


def test_param_counts():
    net = build_conv()
    n_blk, n_tot = models.block_params(net), models.count_params(net)
    print(f"\nConv64: block {n_blk}, total {n_tot} (+{n_tot - models.B_PARAMS} vs B)")
    assert BLOCK_PARAMS == 6956
    assert n_blk == BLOCK_PARAMS and n_tot == models.B_PARAMS + BLOCK_PARAMS
    shapes = {k: tuple(p.shape) for k, p in net.mixer.named_parameters()}
    assert shapes == {"conv1.weight": (H, C, 3, 3), "conv1.bias": (H,), "conv2.weight": (C, H, 3, 3),
                      "conv2.bias": (C,)}


def test_skeleton_equals_B_at_same_seed():
    """weights_init runs on the skeleton before the block exists: same skeleton weights (and RNG use) as B."""
    for seed in (0, 3):
        ref = original_turbnet(seed=seed).state_dict()
        sd = build_conv(seed=seed).state_dict()
        assert set(sd) - set(ref) == {f"mixer.{k}" for k in ("conv1.weight", "conv1.bias", "conv2.weight",
                                                             "conv2.bias")}
        assert all(torch.equal(sd[k], ref[k]) for k in ref)


def test_zero_init_equals_B():
    ref = original_turbnet(seed=0).to(DEV)
    net = build_conv(seed=1).to(DEV)
    assert net.mixer.conv2.weight.abs().sum() == 0 and net.mixer.conv2.bias.abs().sum() == 0
    assert net.mixer.conv1.weight.abs().sum() > 0
    missing, unexpected = net.load_state_dict(ref.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("mixer.") for k in missing)
    x = inputs(device=DEV)
    with torch.no_grad():
        d = (ref.eval()(x) - net.eval()(x)).abs().max().item()
    print(f"\nConv64 at init vs B ({DEV}): max |diff| = {d:.3g}")
    assert d <= 1e-6


def test_gradients_after_two_adam_steps():
    net = build_conv().to(DEV).train()
    opt = torch.optim.Adam(net.parameters(), lr=6e-4, betas=(0.5, 0.999), weight_decay=0.0)
    crit = nn.L1Loss()
    x = inputs(batch=4, device=DEV)
    t = inputs(batch=4, seed=9, device=DEV)
    for step in (1, 2):
        net.zero_grad(set_to_none=True)
        crit(net(x), t).backward()
        nonzero = {k: p.grad is not None and p.grad.abs().sum().item() > 0 for k, p in net.mixer.named_parameters()}
        if step == 1:  # zero last convolution: only it gets a gradient at the first step
            assert {k for k, v in nonzero.items() if v} == {"conv2.weight", "conv2.bias"}, nonzero
        opt.step()
    assert all(nonzero.values()), [k for k, v in nonzero.items() if not v]


def test_locality_5x5():
    """Block alone, last convolution random: perturbing one pixel changes the residual branch exactly in the 5x5
    neighbourhood (Chebyshev radius 2: two 3x3 convolutions) and is exactly 0 elsewhere."""
    torch.manual_seed(0)
    blk = blocks.make_block("Conv", C, 64).eval()
    nn.init.normal_(blk.conv2.weight, std=0.05)
    nn.init.normal_(blk.conv2.bias, std=0.05)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(1, C, 64, 64, generator=g)
    xp = x.clone()
    i0, j0 = 32, 32
    xp[0, :, i0, j0] += torch.randn(C, generator=g)
    ii, jj = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    cheb = torch.maximum((ii - i0).abs(), (jj - j0).abs())
    with torch.no_grad():
        delta = (blk(xp) - blk(x) - (xp - x)).abs().amax(dim=(0, 1))
    near, far = delta[cheb <= 2].min().item(), delta[cheb > 2].max().item()
    print(f"\nConv64 locality: min |delta| in 5x5 {near:.3g}, max outside 5x5 {far:.3g}, "
          f"ring at radius 2 min {delta[cheb == 2].min().item():.3g}")
    assert near > 0 and far == 0.0


def test_config_handling():
    expect = {"variant": "Conv", "level": 64, "channels": 32, "mamba_impl": None}
    assert models.normalize_cfg({"variant": "Conv", "level": 64}) == expect
    assert models.normalize_cfg({"variant": "Conv"}) == expect
    for bad in ({"variant": "Conv", "level": 32}, {"variant": "Conv", "level": 8}, {"variant": "Conv", "levels": "32,64"}):
        with pytest.raises(ValueError):
            models.normalize_cfg(bad)
    run_cfg = {**expect, "levels": None}  # as train.py writes config.json
    assert models.model_cfg_from_run_config(run_cfg) == expect
    assert models.VARIANTS[10] == "Conv" and "Conv" not in models.MAMBA_VARIANTS
