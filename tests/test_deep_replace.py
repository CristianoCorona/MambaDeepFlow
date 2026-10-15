"""Unit tests for the deep-path replacement models TR and DRv11 (mdf/deep_replace.py; docs/lab-notebook.md 2026-10-15,
night 6b).

Run: python -m pytest -s tests/test_deep_replace.py   (needs CUDA for the mamba_ssm kernels)
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mdf import deep_replace, models  # noqa: E402
from test_blocks import build, inputs, needs_cuda, original_turbnet  # noqa: E402

TR_PARAMS = 493315  # B (2 332 035) minus layer4-6 and dlayer6-4 (1 838 720)
MDV128_PARAMS = 169216  # one MDv block at 128 channels (tests/test_mdv.py formula)
DRV11_PARAMS = TR_PARAMS + 11 * MDV128_PARAMS  # 2 354 691


@pytest.fixture
def deterministic_cudnn():
    """Exact comparisons: cuDNN's default algorithms differ run to run at ~1e-8 (two TR passes do too)."""
    old = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    yield
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = old


def build_dr(variant, seed=0):
    torch.manual_seed(seed)
    return models.build_from_cfg({"variant": variant})


@needs_cuda
@pytest.mark.parametrize("variant", ["TR", "DRv11"])
def test_output_shape(variant):
    net = build_dr(variant).cuda()
    assert net(inputs(batch=3, device="cuda")).shape == (3, 3, 128, 128)


@needs_cuda
def test_param_counts():
    tr, dr = build_dr("TR"), build_dr("DRv11")
    print(f"\n  TR {models.count_params(tr)}, DRv11 {models.count_params(dr)} (blocks {models.block_params(dr)}), "
          f"B {models.B_PARAMS}")
    assert models.count_params(tr) == TR_PARAMS
    assert models.block_params(tr) == 0
    assert models.count_params(dr) == DRV11_PARAMS
    assert models.block_params(dr) == 11 * MDV128_PARAMS
    assert len(dr.mixers) == 11
    assert all(models.count_params(b) == MDV128_PARAMS for b in dr.mixers)
    for name in deep_replace.DEEP_LAYERS:
        assert not hasattr(tr, name) and not hasattr(dr, name), name


@needs_cuda
def test_shared_layers_initialised_as_b():
    b, tr, dr = original_turbnet(seed=3), build_dr("TR", seed=3), build_dr("DRv11", seed=3)
    tr_sd, dr_sd = tr.state_dict(), dr.state_dict()
    assert set(tr_sd) == {k for k in b.state_dict() if k.split(".")[0] not in deep_replace.DEEP_LAYERS}
    for k, v in tr_sd.items():
        assert torch.equal(v, b.state_dict()[k]), k
        assert torch.equal(v, dr_sd[k]), k
    assert all(torch.count_nonzero(blk.out_proj.weight) == 0 for blk in dr.mixers)
    # the 11 blocks have their own (different) random weights
    assert not torch.equal(dr.mixers[0].in_proj.weight, dr.mixers[1].in_proj.weight)


@needs_cuda
def test_drv11_equals_tr_at_init(deterministic_cudnn):
    tr, dr = build_dr("TR", seed=1).cuda(), build_dr("DRv11", seed=1).cuda()
    x = inputs(batch=4, device="cuda")
    for mode in ("train", "eval"):
        getattr(tr, mode)(), getattr(dr, mode)()
        with torch.no_grad():
            assert torch.equal(dr(x), tr(x)), mode
    with torch.no_grad():  # eval mode (train mode would update the BatchNorm statistics of dr once more)
        h = dr.layer3(dr.layer2b(dr.layer2(dr.layer1(x.clone()))))
        assert torch.equal(dr.mixers(h), h)  # every block is exactly the identity


@needs_cuda
def test_tr_matches_explicit_forward(deterministic_cudnn):
    """TR's forward against the original modules with dlayer3(cat([out3, out3]))."""
    tr = build_dr("TR").cuda().eval()
    x = inputs(device="cuda")
    with torch.no_grad():
        out1 = tr.layer1(x)
        out2 = tr.layer2(out1)
        out2b = tr.layer2b(out2)
        out3 = tr.layer3(out2b)
        assert out3.shape[1:] == (128, 8, 8)
        d3 = tr.dlayer3(torch.cat([out3, out3], 1))
        d2b = tr.dlayer2b(torch.cat([d3, out2b], 1))
        d2 = tr.dlayer2(torch.cat([d2b, out2], 1))
        ref = tr.dlayer1(torch.cat([d2, out1], 1))
        assert torch.equal(tr(x), ref)


@needs_cuda
def test_gradients_after_two_adam_steps():
    net = build_dr("DRv11").cuda().train()
    opt = torch.optim.Adam(net.parameters(), lr=6e-4, betas=(0.5, 0.999), weight_decay=0.0)
    crit = nn.L1Loss()
    x = inputs(batch=4, device="cuda")
    t = inputs(batch=4, seed=9, device="cuda")
    for step in (1, 2):
        net.zero_grad()
        crit(net(x), t).backward()
        nonzero = {k: p.grad is not None and p.grad.abs().sum().item() > 0 for k, p in net.mixers.named_parameters()}
        if step == 1:  # zero out_proj: only the out_proj of each block gets a gradient at the first step
            assert {k for k, v in nonzero.items() if v} == {f"{i}.out_proj.weight" for i in range(11)}, nonzero
        opt.step()
    assert len(nonzero) == 11 * len(list(net.mixers[0].parameters()))
    assert all(nonzero.values()), [k for k, v in nonzero.items() if not v]


def test_cfg():
    assert models.normalize_cfg({"variant": "TR"}) == {"variant": "TR", "level": None, "channels": 32,
                                                       "mamba_impl": None}
    assert models.normalize_cfg({"variant": "DRv11"}) == {"variant": "DRv11", "level": None, "channels": 32,
                                                          "mamba_impl": "cuda"}
    # config.json of a run -> the same configuration
    run_cfg = {"variant": "DRv11", "level": None, "levels": None, "channels": 32, "mamba_impl": "cuda"}
    assert models.model_cfg_from_run_config(run_cfg) == models.normalize_cfg({"variant": "DRv11"})
    for bad in ({"variant": "TR", "level": 8}, {"variant": "DRv11", "levels": "8,16"},
                {"variant": "DRv11", "channels": 33}):
        with pytest.raises(ValueError):
            models.normalize_cfg(bad)


@needs_cuda
def test_existing_variants_unchanged():
    """Parameter counts and state-dict keys of the existing variants (values as recorded in the notebook)."""
    assert models.count_params(build("B")) == models.B_PARAMS
    b_keys = set(original_turbnet().state_dict())
    for variant, level, kw in (("MD", 8, {}), ("MDv", 8, {}), ("MDv", 64, {}), ("SA", 8, {}), ("CBAM", 8, {})):
        net = build(variant, level, **kw)
        keys = set(net.state_dict())
        assert b_keys <= keys and all(k.startswith("mixer.") for k in keys - b_keys), (variant, level)
        assert models.count_params(net) == models.B_PARAMS + models.block_params(net)
    assert models.block_params(build("MDv", 8)) == MDV128_PARAMS  # same block as in DRv11
    multi = build("MDv", None, levels="8,16,32")
    assert set(multi.mixers) == {"32", "16", "8"}
    assert models.count_params(multi) == models.B_PARAMS + models.block_params(multi)
    assert models.VARIANTS[:6] == ("B", "B-large", "CBAM", "SA", "MD", "MDv")
