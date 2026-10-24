"""Unit tests for the night-9 ladder of one-ingredient ablations of MDv-local (mdf/ablation.SS2DVMambaLocalLadder;
docs/lab-notebook.md 2026-10-17): L-nogate, L-nonorm, L-noprenorm, L-noexp, L-dense, all after layer1 (32 channels,
64x64) as MDv64-local.

Run: python -m pytest -s tests/test_ladder64.py   (needs CUDA, as tests/test_ablation64.py)

Checks per variant: shapes; parameter counts (total, block, used); skeleton weights equal to B at the same seed and
output identical to B at initialisation (zero out_proj); non-zero gradient on every used block parameter after two
Adam steps (and exactly zero on the unused ones / rows); locality (one-pixel perturbation, random out_proj: the block
output changes in the whole 3x3 neighbourhood and exactly nowhere else); difference from MDv64-local in the removed
ingredient only (same weights at the same seed where the ingredient is not involved, and a functional equivalence test
per variant, see test_differs_from_local_only_in_ingredient).
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mdf import ablation, blocks, models  # noqa: E402
from test_blocks import inputs, needs_cuda, original_turbnet  # noqa: E402

NAMES = list(models.LADDER_VARIANTS)
C, NS = 32, 16


def build_v(name, seed=0, **kw):
    torch.manual_seed(seed)
    return models.build_from_cfg({"variant": name, "level": 64, **kw})


def local_block(seed, **kw):
    """The MDv64-local block as TurbNetGMixer creates it (after the skeleton) at this seed."""
    return build_v("MDv-local", seed=seed, **kw).mixer


def counts(name):
    """(block parameters, used block parameters) from the layer shapes."""
    e = C if name == "L-noexp" else 2 * C
    r = math.ceil(C / 16)
    prenorm = 0 if name == "L-noprenorm" else 2 * C
    in_proj = C * 2 * e
    conv = (e * e * 9 if name == "L-dense" else e * 9) + e
    scan = 4 * (r + 2 * NS) * e + 4 * (e * r + e) + 4 * e * NS  # x_proj, dt_proj (w, b), A_log: unused
    ds = 4 * e
    out_norm = 0 if name == "L-nonorm" else 2 * e
    out_proj = e * C
    total = prenorm + in_proj + conv + scan + ds + out_norm + out_proj
    used = total - scan - (C * e if name == "L-nogate" else 0)  # L-nogate: z rows of in_proj unused
    return total, used


def randomise_out(blk, std=0.05):
    nn.init.normal_(blk.out_proj.weight, std=std)  # with a zero out_proj every comparison would be trivial
    return blk


@needs_cuda
@pytest.mark.parametrize("name", NAMES)
def test_shapes(name):
    net = build_v(name).cuda().train()
    assert isinstance(net.mixer, ablation.SS2DVMambaLocalLadder) and net.mix_after == "layer1"
    seen = []
    h = net.mixer.register_forward_hook(lambda m, i, o: seen.append((tuple(i[0].shape), tuple(o.shape))))
    y = net(inputs(device="cuda"))
    h.remove()
    assert y.shape == (2, 3, 128, 128) and torch.isfinite(y).all()
    assert seen == [((2, C, 64, 64), (2, C, 64, 64))], seen


def test_param_counts():
    print()
    local = build_v("MDv-local")
    n_local, u_local = models.block_params(local), ablation.used_param_count(local.mixer)
    print(f"  MDv64-local: block {n_local} (used {u_local}), total {models.count_params(local)}")
    assert (n_local, u_local) == (20800, 7232)
    for name in NAMES:
        net = build_v(name)
        n_blk, n_tot, used = models.block_params(net), models.count_params(net), ablation.used_param_count(net.mixer)
        print(f"  {name}: block {n_blk} (used {used}), total {n_tot} (used total {models.B_PARAMS + used}); "
              f"vs MDv64-local block {n_blk - n_local:+d}, used {used - u_local:+d}")
        assert (n_blk, used) == counts(name)
        assert n_tot == models.B_PARAMS + n_blk
    assert counts("L-noexp") == (10432, 3648) and counts("L-dense") == (57088, 43520)


@pytest.mark.parametrize("name", NAMES)
def test_skeleton_equals_B_at_same_seed(name):
    for seed in (0, 3):
        ref = original_turbnet(seed=seed).state_dict()
        sd = build_v(name, seed=seed).state_dict()
        assert all(k.startswith("mixer.") for k in set(sd) - set(ref))
        assert all(torch.equal(sd[k], ref[k]) for k in ref)


@needs_cuda
@pytest.mark.parametrize("name", NAMES)
def test_zero_init_equals_B(name):
    for impl, device in (("cuda", "cuda"), ("ref", "cpu")):
        ref = original_turbnet(seed=0).to(device)
        net = build_v(name, seed=1, mamba_impl=impl).to(device)
        assert net.mixer.out_proj.weight.abs().sum() == 0
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
    rows = net.mixer.unused_rows()
    assert unused == set(ablation.UNUSED_LOCAL) and (bool(rows) == (name == "L-nogate"))
    for step in (1, 2):
        net.zero_grad(set_to_none=True)
        crit(net(x), t).backward()
        params = dict(net.mixer.named_parameters())
        assert all(params[k].grad is None for k in unused), [k for k in unused if params[k].grad is not None]
        nonzero = {}
        for k, p in params.items():
            if k in unused:
                continue
            g = p.grad
            if k in rows:  # used rows must move, unused rows get exactly zero
                assert g[rows[k]].abs().sum().item() == 0
                keep = torch.ones(g.shape[0], dtype=torch.bool, device=g.device)
                keep[rows[k]] = False
                g = g[keep]
            nonzero[k] = g is not None and g.abs().sum().item() > 0
        if step == 1:  # zero out_proj: only out_proj gets a gradient at the first step
            assert {k for k, v in nonzero.items() if v} == {"out_proj.weight"}, nonzero
        opt.step()
    assert all(nonzero.values()), [k for k, v in nonzero.items() if not v]


@needs_cuda
@pytest.mark.parametrize("name", NAMES)
def test_locality_3x3(name):
    """Block alone, random out_proj: perturbing one pixel changes the mixing term (block output minus the residual)
    at every pixel of the 3x3 neighbourhood and is exactly 0 elsewhere."""
    torch.manual_seed(0)
    blk = randomise_out(blocks.make_block(name, C, 64)).cuda().eval()
    g = torch.Generator().manual_seed(5)
    x = torch.randn(1, C, 64, 64, generator=g).cuda()
    xp = x.clone()
    i0, j0 = 32, 32
    xp[0, :, i0, j0] += torch.randn(C, generator=g).cuda()
    ii, jj = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    cheb = torch.maximum((ii - i0).abs(), (jj - j0).abs()).cuda()
    with torch.no_grad():
        delta = (blk(xp) - blk(x) - (xp - x)).abs().amax(dim=(0, 1))
    near, far = delta[cheb <= 1].min().item(), delta[cheb > 1].max().item()
    print(f"\n{name} locality: min |delta| in 3x3 {near:.3g}, max outside 3x3 {far:.3g}")
    assert near > 0 and far == 0.0


REMOVED = {"L-nogate": set(), "L-nonorm": {"out_norm.weight", "out_norm.bias"},
           "L-noprenorm": {"norm.weight", "norm.bias"}, "L-dense": set()}
CHANGED = {"L-nogate": set(), "L-nonorm": set(), "L-noprenorm": set(), "L-dense": {"conv2d.weight"}}


@pytest.mark.parametrize("name", [n for n in NAMES if n != "L-noexp"])
def test_same_weights_as_local(name):
    """At a given seed every block weight not involved in the removed ingredient equals MDv64-local's (same keys, same
    order, minus the removed LayerNorm; L-dense: only conv2d.weight differs, in shape; its bias is drawn after all the
    other weights, so it differs in value)."""
    for seed in (0, 3):
        ref = local_block(seed).state_dict()
        sd = build_v(name, seed=seed).mixer.state_dict()
        assert list(sd) == [k for k in ref if k not in REMOVED[name]]
        changed = CHANGED[name] | ({"conv2d.bias"} if name == "L-dense" else set())
        assert all(torch.equal(sd[k], ref[k]) for k in sd if k not in changed)
        if name == "L-dense":
            assert sd["conv2d.weight"].shape == (64, 64, 3, 3) and ref["conv2d.weight"].shape == (64, 1, 3, 3)


def test_noexp_is_local_with_expand_1():
    """L-noexp cannot share weights with MDv64-local (E = 32 instead of 64 changes every inner shape and the RNG
    draws). Equivalent check: it is exactly SS2DVMambaLocal(32, expand=1), same weights at the same seed and same
    output; the only constructor argument that differs from MDv64-local is expand."""
    for seed in (0, 3):
        torch.manual_seed(seed)
        ref = ablation.SS2DVMambaLocal(C, "ref", expand=1)
        torch.manual_seed(seed)
        blk = blocks.make_block("L-noexp", C, 64, "ref")
        sd, rsd = blk.state_dict(), ref.state_dict()
        assert list(sd) == list(rsd) and all(torch.equal(sd[k], rsd[k]) for k in rsd)
        assert list(sd) == list(local_block(seed, mamba_impl="ref").state_dict())  # same layers, other shapes
        assert blk.d_inner == C and local_block(seed).d_inner == 2 * C
        randomise_out(ref)
        blk.load_state_dict(ref.state_dict())
        x = torch.randn(2, C, 64, 64, generator=torch.Generator().manual_seed(seed))
        with torch.no_grad():
            assert torch.equal(blk(x), ref(x))


def test_differs_from_local_only_in_ingredient():
    """Functional check on CPU, random out_proj, MDv64-local and the variant with the same weights:
    - L-nonorm / L-noprenorm: equal to MDv64-local with that LayerNorm replaced by nn.Identity;
    - L-nogate: the input of out_proj times SiLU(z) (z from MDv64-local's in_proj) equals MDv64-local's input of
      out_proj;
    - L-dense: with its dense kernel set to the depthwise kernel on the diagonal (zero off the diagonal) and the same
      bias, it equals MDv64-local up to float rounding;
    and in every case the variant differs from MDv64-local itself (the ingredient does something)."""
    g = torch.Generator().manual_seed(7)
    x = torch.randn(2, C, 64, 64, generator=g)
    for name in ("L-nogate", "L-nonorm", "L-noprenorm", "L-dense"):
        local = randomise_out(local_block(0, mamba_impl="ref")).eval()
        torch.manual_seed(0)
        blk = blocks.make_block(name, C, 64, "ref").eval()
        sd = {k: v for k, v in local.state_dict().items() if k not in REMOVED[name]}
        if name == "L-dense":
            sd["conv2d.weight"] = blk.conv2d.weight.detach().clone()  # its own dense kernel (default init)
        blk.load_state_dict(sd, strict=True)
        with torch.no_grad():
            assert (local(x) - blk(x)).abs().max().item() > 1e-4, name
            if name == "L-dense":
                dense = torch.zeros(64, 64, 3, 3)
                dense[torch.arange(64), torch.arange(64)] = local.conv2d.weight[:, 0]
                blk.conv2d.weight.copy_(dense)
            y = blk(x)
            if name in ("L-nonorm", "L-noprenorm"):
                setattr(local, "out_norm" if name == "L-nonorm" else "norm", nn.Identity())
                assert torch.equal(local(x), y), name
            elif name == "L-nogate":
                seen = {}
                for tag, m in (("local", local), ("nogate", blk)):
                    m.out_proj.register_forward_hook(lambda mod, i, o, tag=tag: seen.__setitem__(tag, i[0]))
                local(x), blk(x)
                z = local.in_proj(local.norm(x.permute(0, 2, 3, 1))).chunk(2, dim=-1)[1]
                assert torch.equal(seen["nogate"] * F.silu(z), seen["local"])
            else:
                d = (local(x) - y).abs().max().item()
                print(f"\nL-dense with diagonal kernel vs MDv64-local: max |diff| {d:.3g}")
                assert d < 1e-5


def test_config_handling():
    for name in NAMES:
        expect = {"variant": name, "level": 64, "channels": 32, "mamba_impl": "cuda"}
        assert models.normalize_cfg({"variant": name, "level": 64}) == expect
        assert models.normalize_cfg({"variant": name}) == expect
        for bad in ({"variant": name, "level": 32}, {"variant": name, "levels": "32,64"}):
            with pytest.raises(ValueError):
                models.normalize_cfg(bad)
        run_cfg = {**expect, "levels": None}  # as train.py writes config.json
        assert models.model_cfg_from_run_config(run_cfg) == expect
    assert models.VARIANTS[-5:] == models.LADDER_VARIANTS and models.VARIANTS[10] == "Conv"
    with pytest.raises(ValueError):
        ablation.SS2DVMambaLocalLadder(C, "L-unknown")
