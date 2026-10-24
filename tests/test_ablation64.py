"""Unit tests for the night-8 ablations of MDv64 (mdf/ablation.py; docs/lab-notebook.md 2026-10-16):
MDv-local64 (no propagation along the scan), SA64 (existing SA block at 64x64), MDv-dec64 (MDv block after dlayer2),
MDv16 (existing MDv block at 16x16).

Run: python -m pytest -s tests/test_ablation64.py   (needs CUDA for the mamba_ssm kernels)

Checks per variant: shapes, parameter counts (total, block, used), output identical to B at initialisation (same
skeleton weights, zero output projection), non-zero gradient on every used block parameter after two Adam steps.
MDv-local: explicit locality test (a one-pixel perturbation changes the mixer output only in the 3x3 neighbourhood,
while MDv64 changes it far away too) and same weights as MDv64 at the same seed. SA64: efficient SDPA backend and
peak memory.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mdf import ablation, blocks, models  # noqa: E402
from test_blocks import build, inputs, needs_cuda, original_turbnet  # noqa: E402

# name -> (cfg, block channels, block map size)
VARIANTS = {
    "MDv-local64": ({"variant": "MDv-local", "level": 64}, 32, 64),
    "SA64": ({"variant": "SA", "level": 64}, 32, 64),
    "MDv-dec64": ({"variant": "MDv-dec", "level": 64}, 32, 64),
    "MDv16": ({"variant": "MDv", "level": 16}, 64, 16),
}
NAMES = list(VARIANTS)


def build_v(name, seed=0, **kw):
    torch.manual_seed(seed)
    return models.build_from_cfg({**VARIANTS[name][0], **kw})


def mdv_block_params(c, ns=16):
    e, r = 2 * c, math.ceil(c / 16)
    return (2 * c + c * 2 * e + e * 9 + e + 4 * (r + 2 * ns) * e + 4 * (e * r + e) + 4 * e * ns + 4 * e
            + 2 * e + e * c)


def expected_block_params(name):
    _, c, _ = VARIANTS[name]
    if name == "SA64":
        return 2 * c + (c * 3 * c + 3 * c) + (c * c + c)  # LayerNorm, qkv, out
    return mdv_block_params(c)


def used_block_params(name):
    _, c, _ = VARIANTS[name]
    if name != "MDv-local64":
        return expected_block_params(name)
    e, r, ns = 2 * c, math.ceil(c / 16), 16
    return expected_block_params(name) - (4 * (r + 2 * ns) * e + 4 * (e * r + e) + 4 * e * ns)


def randomise_out(blk, std=0.05):
    w = blk.out.weight if isinstance(blk, blocks.SelfAttention2D) else blk.out_proj.weight
    nn.init.normal_(w, std=std)  # with a zero output projection every comparison would be trivial
    return blk


@needs_cuda
@pytest.mark.parametrize("name", NAMES)
def test_shapes(name):
    net = build_v(name).cuda().train()
    _, c, s = VARIANTS[name]
    seen = []
    h = net.mixer.register_forward_hook(lambda m, i, o: seen.append((tuple(i[0].shape), tuple(o.shape))))
    y = net(inputs(device="cuda"))
    h.remove()
    assert y.shape == (2, 3, 128, 128) and torch.isfinite(y).all()
    assert seen == [((2, c, s, s), (2, c, s, s))], seen


@needs_cuda
def test_param_counts():
    print()
    for name in NAMES:
        net = build_v(name)
        n_blk, n_tot = models.block_params(net), models.count_params(net)
        print(f"  {name}: block {n_blk} (used {used_block_params(name)}), total {n_tot} "
              f"(+{n_tot - models.B_PARAMS} vs B; used total {models.B_PARAMS + used_block_params(name)})")
        assert n_blk == expected_block_params(name)
        assert n_tot == models.B_PARAMS + n_blk


@needs_cuda
@pytest.mark.parametrize("name", NAMES)
def test_zero_init_equals_B(name):
    impls = [("cuda", "cuda")] + ([("ref", "cpu")] if name != "SA64" else [("cuda", "cpu")])
    for impl, device in impls:
        kw = {} if name == "SA64" else {"mamba_impl": impl}
        ref = original_turbnet(seed=0).to(device)
        net = build_v(name, seed=1, **kw).to(device)
        missing, unexpected = net.load_state_dict(ref.state_dict(), strict=False)
        assert not unexpected and missing and all(k.startswith("mixer.") for k in missing)
        x = inputs(device=device)
        with torch.no_grad():
            d = (ref.eval()(x) - net.eval()(x)).abs().max().item()
        print(f"\n{name} ({impl}, {device}) at init vs B: max |diff| = {d:.3g}")
        assert d <= 1e-6


@needs_cuda
@pytest.mark.parametrize("name", NAMES)
def test_gradients_after_two_adam_steps(name):
    net = build_v(name).cuda().train()
    opt = torch.optim.Adam(net.parameters(), lr=6e-4, betas=(0.5, 0.999), weight_decay=0.0)
    crit = nn.L1Loss()
    x = inputs(batch=4, device="cuda")
    t = inputs(batch=4, seed=9, device="cuda")
    unused = ablation.unused_param_names(net.mixer)
    first = {"out.weight", "out.bias"} if name == "SA64" else {"out_proj.weight"}
    for step in (1, 2):
        net.zero_grad(set_to_none=True)
        crit(net(x), t).backward()
        grads = dict(net.mixer.named_parameters())
        nonzero = {k: p.grad is not None and p.grad.abs().sum().item() > 0 for k, p in grads.items() if k not in unused}
        assert all(grads[k].grad is None for k in unused), [k for k in unused if grads[k].grad is not None]
        if step == 1:  # zero output projection: only the projection itself gets a gradient at the first step
            assert {k for k, v in nonzero.items() if v} == first, nonzero
        opt.step()
    assert all(nonzero.values()), [k for k, v in nonzero.items() if not v]
    n_used = sum(p.numel() for k, p in net.mixer.named_parameters() if k not in unused)
    print(f"\n{name}: used block params {n_used}, unused {sorted(unused)}")
    assert n_used == used_block_params(name)


@needs_cuda
def test_same_weights_as_mdv64():
    """MDv-local64 and MDv-dec64 differ from MDv64 only in the forward / position: same initial weights per seed."""
    for seed in (0, 3):
        ref = build("MDv", 64, seed=seed).state_dict()
        local = build_v("MDv-local64", seed=seed).state_dict()
        dec = build_v("MDv-dec64", seed=seed).state_dict()
        assert list(local) == list(ref) and all(torch.equal(local[k], ref[k]) for k in ref)
        assert list(dec) == list(ref) and all(torch.equal(dec[k], ref[k]) for k in ref)


def strong_memory(blk, dt=0.5, a=1e-3):
    """Test-only parameter change: slow decay (A = -a) and large steps (dt) in every direction, so that the
    propagated signal is far above float32 rounding (at the upstream init it is ~1e-4 of the local one)."""
    with torch.no_grad():
        blk.A_logs.fill_(math.log(a))
        blk.dt_projs_weight.zero_()
        blk.dt_projs_bias.fill_(dt + math.log(-math.expm1(-dt)))  # inverse softplus
    return blk


@needs_cuda
@pytest.mark.parametrize("memory", ["init", "strong"])
def test_locality(memory):
    """Mixer alone (out_proj random): perturb one pixel; MDv-local changes only its 3x3 neighbourhood (exactly 0
    elsewhere), MDv64 with the same weights also changes pixels far away. The CUDA kernels are deterministic in the
    forward, so a pixel whose inputs and scan states are bit-identical gives a bit-identical output: any non-zero
    change is caused by the perturbation. One pixel among 4096 moves the far outputs only a little (one term of the
    scan sums; at the upstream init ~1e-7 against ~1e-3 locally): "init" upstream initialisation, "strong" slow decay
    and large steps (strong_memory), where far changes reach ~1e-5."""
    torch.manual_seed(0)
    mdv = randomise_out(blocks.SS2DVMamba(32, "cuda")).cuda().eval()
    if memory == "strong":
        strong_memory(mdv)
    local = blocks.make_block("MDv-local", 32, 64).cuda().eval()
    local.load_state_dict(mdv.state_dict())
    g = torch.Generator().manual_seed(5)
    x = torch.randn(1, 32, 64, 64, generator=g).cuda()
    xp = x.clone()
    i0, j0 = 32, 32
    xp[0, :, i0, j0] += torch.randn(32, generator=g).cuda()
    ii, jj = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    cheb = torch.maximum((ii - i0).abs(), (jj - j0).abs()).cuda()
    far_pts = ((32, 0), (0, 32), (0, 0), (63, 63), (32, 63), (63, 32))
    print()
    with torch.no_grad():
        for name, blk in (("MDv-local64", local), ("MDv64", mdv)):
            delta = (blk(xp) - blk(x) - (xp - x)).abs().amax(dim=(0, 1))  # mixing term only, max over channels
            near, far = delta[cheb <= 1].min().item(), delta[cheb > 1].max().item()
            far16 = delta[cheb >= 16]
            print(f"  [{memory}] {name}: min |delta| in 3x3 {near:.3g}, max outside 3x3 {far:.3g}; pixels >= 16 px "
                  f"away: max {far16.max().item():.3g}, share changed {(far16 > 0).float().mean().item():.1%}; "
                  + ", ".join(f"{p} {delta[p].item():.2g}" for p in far_pts))
            if name == "MDv-local64":
                assert near > 0 and far == 0.0
            else:
                assert near > 0 and (far16 > 0).float().mean().item() > (0.99 if memory == "strong" else 0.5)
                if memory == "strong":
                    for p in far_pts:
                        assert delta[p].item() > 1e-7, p


@needs_cuda
def test_sa64_sdpa_backend_and_memory():
    """SA64 in float32: flash attention is not eligible, the default picks the memory-efficient kernel (checked with
    the profiler and by restricting SDPA to it); training step of the full network at batch 10 fits in memory."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.profiler import ProfilerActivity, profile

    def train_step(net, x, t):
        net.zero_grad(set_to_none=True)
        y = net(x)
        nn.L1Loss()(y, t).backward()
        return y.detach()

    net = build_v("SA64").cuda().train()
    randomise_out(net.mixer)
    x, t = inputs(batch=10, device="cuda"), inputs(batch=10, seed=9, device="cuda")
    train_step(net, x, t)  # warm-up
    torch.cuda.synchronize()
    peaks = {}
    for label, ctx in (("default", None), ("efficient only", SDPBackend.EFFICIENT_ATTENTION)):
        torch.cuda.reset_peak_memory_stats()
        if ctx is None:
            y = train_step(net, x, t)
            grads = [p.grad.clone() for p in net.mixer.parameters()]
        else:
            with sdpa_kernel(ctx):
                y2 = train_step(net, x, t)
                grads2 = [p.grad.clone() for p in net.mixer.parameters()]
        torch.cuda.synchronize()
        peaks[label] = torch.cuda.max_memory_allocated() / 2 ** 20
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        train_step(net, x, t)
        torch.cuda.synchronize()
    names = {e.key for e in prof.key_averages()}
    attn = sorted(n for n in names if "attention" in n.lower())
    print(f"\nSA64 training step batch 10: peak alloc {peaks} MiB; attention ops: {attn}")
    torch.testing.assert_close(y, y2, rtol=1e-5, atol=1e-6)  # cuDNN / atomics: float32 rounding only
    for a, b in zip(grads, grads2):  # the efficient backward accumulates with atomics: run-to-run rounding
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3 * a.abs().max().item())
    assert any("efficient_attention" in n for n in attn) and not any("flash" in n for n in attn)
    assert max(peaks.values()) < 7 * 1024


def test_config_handling():
    assert models.normalize_cfg({"variant": "MDv-local", "level": 64}) == \
        {"variant": "MDv-local", "level": 64, "channels": 32, "mamba_impl": "cuda"}
    assert models.normalize_cfg({"variant": "MDv-dec"}) == \
        {"variant": "MDv-dec", "level": 64, "channels": 32, "mamba_impl": "cuda"}
    for bad in ({"variant": "MDv-dec", "level": 32}, {"variant": "MDv-local", "levels": "8,16"},
                {"variant": "MDv-dec", "levels": "32,64"}):
        with pytest.raises(ValueError):
            models.normalize_cfg(bad)
    for cfg in ({"variant": "MDv-local", "level": 64}, {"variant": "MDv-dec", "level": 64}):
        run_cfg = {**models.normalize_cfg(cfg), "levels": None}  # as train.py writes config.json
        assert models.model_cfg_from_run_config(run_cfg) == models.normalize_cfg(cfg)
