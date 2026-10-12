"""Unit tests for the global-mixing blocks and the model factory (docs/lab-notebook.md, 2026-10-12).

Run: python -m pytest -s tests/test_blocks.py   (needs CUDA for the mamba_ssm kernels)

Exact-equality tests run in eval() mode on the CPU where possible (deterministic kernels); tests that
need the fused Mamba kernels run on the GPU with an explicit tolerance.
"""

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mdf import blocks, models  # noqa: E402
from mdf.mamba_ref import selective_scan_ref as my_scan_ref  # noqa: E402

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="CUDA required")
DEV = "cuda" if CUDA else "cpu"

# float32 tolerances for fused CUDA kernels vs pure-PyTorch reference: the values used by the official
# mamba_ssm tests for float32 (tests/ops/test_selective_scan.py: rtol 6e-4, atol 2e-3)
RTOL, ATOL = 6e-4, 2e-3


def build(variant, level=8, seed=0, **kw):
    torch.manual_seed(seed)
    return models.build_from_cfg({"variant": variant, "level": level, **kw})


def original_turbnet(seed=0, expo=5):
    DfpNet, _ = models.original_modules()
    torch.manual_seed(seed)
    net = DfpNet.TurbNetG(channelExponent=expo, dropout=0.0)
    net.apply(DfpNet.weights_init)
    return net


def skeleton_state(net):
    return {k: v for k, v in net.state_dict().items() if not k.startswith("mixer.")}


def inputs(batch=2, seed=123, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 3, 128, 128, generator=g).to(device)


# 1 -------------------------------------------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("variant", ["B", "B-large", "CBAM", "SA", "MD"])
@pytest.mark.parametrize("level", models.LEVELS)
def test_shapes(variant, level):
    if variant in ("B", "B-large") and level != 8:
        pytest.skip("no block: level irrelevant")
    net = build(variant, level).to(DEV).train()
    out = net(inputs(device=DEV))
    assert out.shape == (2, 3, 128, 128)
    assert torch.isfinite(out).all()


# 2 -------------------------------------------------------------------------------------------------------------
def test_B_equals_original():
    ref = original_turbnet(seed=0)
    net = build("B", seed=0)
    assert models.count_params(net) == models.count_params(ref) == models.B_PARAMS == 2332035
    assert type(net) is type(ref)
    # same seed -> same initial weights (B from the factory consumes the RNG as the original)
    for (k1, v1), (k2, v2) in zip(ref.state_dict().items(), net.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2), k1
    # same weights -> identical output (also with different initial weights, after loading)
    net2 = build("B", seed=7)
    net2.load_state_dict(ref.state_dict())
    x = inputs()
    with torch.no_grad():
        assert torch.equal(ref.eval()(x), net2.eval()(x))


def test_B_large_channels():
    net = build("B-large")
    assert net.layer1.layer1_conv.out_channels == 33
    print(f"\nB-large trainable params: {models.count_params(net)}")


# 3 -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("level", models.LEVELS)
def test_identity_block_equals_original(level):
    ref = original_turbnet(seed=0)
    net = build("identity", level, seed=1)
    net.load_state_dict(ref.state_dict())  # strict: the identity block has no parameters
    x = inputs()
    with torch.no_grad():
        assert torch.equal(ref.eval()(x), net.eval()(x))
    ref.train(), net.train()
    assert torch.equal(ref(x), net(x))  # train mode (batch statistics) as well


# 4 -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("variant,impl,device", [
    ("SA", None, "cpu"),
    pytest.param("MD", "cuda", "cuda", marks=needs_cuda),
    pytest.param("MD", "ref", "cpu", marks=needs_cuda),  # building Mamba imports the CUDA extension
])
@pytest.mark.parametrize("level", [8, 32])
def test_zero_init_equals_B(variant, impl, device, level):
    ref = original_turbnet(seed=0).to(device)
    net = build(variant, level, seed=1, mamba_impl=impl).to(device)
    missing, unexpected = net.load_state_dict(ref.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("mixer.") for k in missing)
    x = inputs(device=device)
    with torch.no_grad():
        d = (ref.eval()(x) - net.eval()(x)).abs().max().item()
    print(f"\n{variant}{level} ({impl}, {device}) at init vs B: max |diff| = {d:.3g}")
    assert d <= 1e-6


# 5 -------------------------------------------------------------------------------------------------------------
def test_scan_unscan():
    x = torch.randn(2, 5, 7, 3)  # non-square on purpose (H != W)
    seqs = []
    for k in range(4):
        s = blocks.scan(x, k)
        assert s.shape == (2, 35, 3)
        assert torch.equal(blocks.unscan(s, k, 5, 7), x)
        seqs.append(s)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.equal(seqs[i], seqs[j]), (blocks.SCAN_ORDERS[i], blocks.SCAN_ORDERS[j])
    # explicit order on a 2x3 grid with values = token index in row-major order
    g = torch.arange(6.0).reshape(1, 2, 3, 1)
    assert blocks.scan(g, 0).flatten().tolist() == [0, 1, 2, 3, 4, 5]
    assert blocks.scan(g, 1).flatten().tolist() == [5, 4, 3, 2, 1, 0]
    assert blocks.scan(g, 2).flatten().tolist() == [0, 3, 1, 4, 2, 5]
    assert blocks.scan(g, 3).flatten().tolist() == [5, 2, 4, 1, 3, 0]


# 6 -------------------------------------------------------------------------------------------------------------
@needs_cuda
def test_selective_scan_cuda_vs_ref():
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref as upstream_ref

    torch.manual_seed(0)
    b, d, n, l = 4, 32, 16, 256
    mk = lambda *s, scale=1.0: (scale * torch.randn(*s, device="cuda")).requires_grad_()  # noqa: E731
    u, delta, z = mk(b, d, l), mk(b, d, l, scale=0.5), mk(b, d, l)
    A = (-0.5 * torch.rand(d, n, device="cuda")).requires_grad_()
    B, C = mk(b, n, l), mk(b, n, l)
    D, bias = mk(d), mk(d, scale=0.5)
    args = (u, delta, A, B, C, D, z, bias, True)
    out_cuda = selective_scan_fn(*args)
    out_ref = my_scan_ref(u, delta, A, B, C, D, z=z, delta_bias=bias, delta_softplus=True)
    out_up = upstream_ref(*args)
    assert torch.equal(out_ref, out_up)  # vendored copy == upstream reference
    g = torch.randn_like(out_ref)
    grads_cuda = torch.autograd.grad(out_cuda, [u, delta, A, B, C, D, z, bias], g)
    grads_ref = torch.autograd.grad(out_ref, [u, delta, A, B, C, D, z, bias], g)
    print(f"\nselective_scan cuda vs ref: out max |diff| {(out_cuda - out_ref).abs().max().item():.3g}")
    torch.testing.assert_close(out_cuda, out_ref, rtol=RTOL, atol=ATOL)
    for name, gc, gr in zip("u delta A B C D z bias".split(), grads_cuda, grads_ref):
        print(f"  grad {name}: max |diff| {(gc - gr).abs().max().item():.3g} (max |ref| {gr.abs().max().item():.3g})")
        torch.testing.assert_close(gc, gr, rtol=RTOL, atol=ATOL)


@needs_cuda
@pytest.mark.parametrize("level,channels", [(8, 128), (32, 64)])
def test_md_block_cuda_vs_ref(level, channels):
    torch.manual_seed(0)
    blk = blocks.SS2DMamba(channels, "cuda").cuda()
    # with out_proj = 0 both paths return x exactly and the comparison would be trivial
    nn.init.normal_(blk.mamba.out_proj.weight, std=0.05)
    ref = copy.deepcopy(blk)
    ref.mamba_impl = "ref"
    x = torch.randn(3, channels, level, level, device="cuda")
    x1, x2 = x.clone().requires_grad_(), x.clone().requires_grad_()
    y1, y2 = blk(x1), ref(x2)
    m1, m2 = y1 - x, y2 - x  # the mixing term
    print(f"\nMD{level} cuda vs ref: mixing term max |diff| {(m1 - m2).abs().max().item():.3g} "
          f"(max |term| {m2.abs().max().item():.3g})")
    torch.testing.assert_close(m1, m2, rtol=RTOL, atol=ATOL)
    g = torch.randn_like(y1)
    (y1 * g).sum().backward()
    (y2 * g).sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad, rtol=RTOL, atol=ATOL)
    for (k, p1), (_, p2) in zip(blk.named_parameters(), ref.named_parameters()):
        scale = p2.grad.abs().max().item()
        print(f"  grad {k}: max |diff| {(p1.grad - p2.grad).abs().max().item():.3g} (max |ref| {scale:.3g})")
        torch.testing.assert_close(p1.grad, p2.grad, rtol=RTOL, atol=ATOL * max(1.0, scale))


# 7 -------------------------------------------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("variant", ["SA", "MD", "CBAM"])
@pytest.mark.parametrize("level", [8, 32])
def test_block_gradients_after_two_adam_steps(variant, level):
    net = build(variant, level).cuda().train()
    opt = torch.optim.Adam(net.parameters(), lr=6e-4, betas=(0.5, 0.999), weight_decay=0.0)
    crit = nn.L1Loss()
    x = inputs(batch=4, device="cuda")
    t = inputs(batch=4, seed=9, device="cuda")
    out_names = {"SA": ("out.weight", "out.bias"), "MD": ("mamba.out_proj.weight",), "CBAM": ()}[variant]
    for step in (1, 2):
        net.zero_grad()
        crit(net(x), t).backward()
        nonzero = {k: p.grad is not None and p.grad.abs().sum().item() > 0 for k, p in net.mixer.named_parameters()}
        if step == 1 and variant in ("SA", "MD"):
            # Expected: at the first step the output projection is zero, so no gradient flows to the
            # parameters before it; only the output projection itself receives a gradient.
            assert {k for k, v in nonzero.items() if v} == set(out_names), nonzero
        opt.step()
    assert all(nonzero.values()), [k for k, v in nonzero.items() if not v]


# 8 -------------------------------------------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("variant", ["SA", "MD", "CBAM"])
@pytest.mark.parametrize("level", [8, 32])
def test_weights_init_does_not_touch_block(variant, level):
    """Rebuild the block with the RNG in the same state as inside the factory (seed -> skeleton +
    weights_init -> block) and compare; the skeleton must equal B with the same seed. Negative
    control: applying weights_init to the block does change it (MD conv1d, CBAM conv), so the
    comparison would detect it. SA has no Conv/BatchNorm, so weights_init would not change it anyway."""
    DfpNet, _ = models.original_modules()
    net = build(variant, level, seed=3)
    b = build("B", seed=3)
    for k, v in b.state_dict().items():
        assert torch.equal(v, net.state_dict()[k]), k
    torch.manual_seed(3)
    skel = DfpNet.TurbNetG(channelExponent=5)
    skel.apply(DfpNet.weights_init)
    mult = models.LEVEL_LAYER[level][1]
    fresh = blocks.make_block(variant, 32 * mult, level)
    for k, v in fresh.state_dict().items():
        assert torch.equal(v, net.mixer.state_dict()[k]), k
    if variant == "MD":
        assert torch.count_nonzero(net.mixer.mamba.out_proj.weight) == 0
    if variant == "SA":
        assert torch.count_nonzero(net.mixer.out.weight) == 0 and torch.count_nonzero(net.mixer.out.bias) == 0
    if variant in ("MD", "CBAM"):
        touched = copy.deepcopy(fresh)
        touched.apply(DfpNet.weights_init)
        assert any(not torch.equal(v, touched.state_dict()[k]) for k, v in fresh.state_dict().items())


# 9 -------------------------------------------------------------------------------------------------------------
@needs_cuda
def test_param_counts_and_pe():
    estimates = {("SA", 8): 66e3, ("SA", 32): 17e3, ("MD", 8): 117e3, ("MD", 32): 33e3, ("CBAM", 8): 2e3}
    print()
    for (variant, level), est in estimates.items():
        net = build(variant, level)
        n = models.block_params(net)
        print(f"  {variant}{level}: block params {n} (estimate ~{est:.0f}), total {models.count_params(net)}")
        assert abs(n - est) <= 0.15 * est  # estimates are rounded ("~2k" for CBAM: 2282 is +14 %)
    sa = build("SA", 8).mixer
    names = [k for k, _ in sa.named_parameters()]
    assert not any("pe" in k for k in names)
    assert "pe" not in sa.state_dict()
    assert sa.pe.shape == (64, 128) and not sa.pe.requires_grad
    print(f"  CBAM32: block params {models.block_params(build('CBAM', 32))}")
