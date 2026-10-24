"""Unit tests for the faithful-SS2D block (variant MDv, docs/lab-notebook.md 2026-10-14).

Run: python -m pytest -s tests/test_mdv.py   (needs CUDA for the mamba_ssm kernels)

Same checks as tests/test_blocks.py for MD (shapes, identity at initialisation, CUDA vs reference scan, gradients
after two Adam steps, weights_init does not touch the block, parameter count), plus a fidelity test against a direct
transcription of the official SS2D (below, used only here).
"""

import copy
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mdf import blocks, models  # noqa: E402
from test_blocks import ATOL, RTOL, build, inputs, needs_cuda, original_turbnet  # noqa: E402


# --- Official SS2D, transcribed for reference (tests only) ------------------------------------------------------
# Source: https://github.com/MzeroMiko/VMamba, commit 9ca24aee4bbffdb34619151afe8aa7a6ca278e44 (2024-01-19),
# classification/models/vmamba/vmamba.py, class SS2D (__init__, dt_init, A_log_init, D_init, forward_core,
# forward), Copyright (c) the VMamba authors (Liu et al., arXiv 2401.10166); the repository is MIT-licensed
# (no LICENSE file at that commit). Copied verbatim except: factory_kwargs / device / dtype arguments dropped,
# `selective_scan_fn` imported lazily, `**kwargs` of forward dropped, dropout unused (0).
class OfficialSS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, dropout=0., conv_bias=True, bias=False):
        from einops import repeat  # noqa: F401  (used by A_log_init / D_init)
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner,
                                bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2)
        self.act = nn.SiLU()
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        from einops import repeat
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        from einops import repeat
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
# --- end of the transcription -----------------------------------------------------------------------------------


def randomise_out_proj(blk, std=0.05):
    nn.init.normal_(blk.out_proj.weight, std=std)  # with out_proj = 0 every comparison would be trivial
    return blk


@needs_cuda
@pytest.mark.parametrize("level", [8, 32])
def test_shapes(level):
    net = build("MDv", level).cuda().train()
    y = net(inputs(device="cuda"))
    assert y.shape == (2, 3, 128, 128) and torch.isfinite(y).all()


@needs_cuda
@pytest.mark.parametrize("impl,device", [("cuda", "cuda"), ("ref", "cpu")])
@pytest.mark.parametrize("level", [8, 32])
def test_zero_init_equals_B(impl, device, level):
    ref = original_turbnet(seed=0).to(device)
    net = build("MDv", level, seed=1, mamba_impl=impl).to(device)
    missing, unexpected = net.load_state_dict(ref.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("mixer.") for k in missing)
    x = inputs(device=device)
    with torch.no_grad():
        d = (ref.eval()(x) - net.eval()(x)).abs().max().item()
    print(f"\nMDv{level} ({impl}, {device}) at init vs B: max |diff| = {d:.3g}")
    assert d <= 1e-6


@needs_cuda
@pytest.mark.parametrize("level,channels", [(8, 128), (32, 64)])
def test_cuda_vs_ref(level, channels):
    torch.manual_seed(0)
    blk = randomise_out_proj(blocks.SS2DVMamba(channels, "cuda")).cuda()
    ref = copy.deepcopy(blk)
    ref.mamba_impl = "ref"
    x = torch.randn(3, channels, level, level, device="cuda")
    x1, x2 = x.clone().requires_grad_(), x.clone().requires_grad_()
    y1, y2 = blk(x1), ref(x2)
    m1, m2 = y1 - x, y2 - x
    print(f"\nMDv{level} cuda vs ref: mixing term max |diff| {(m1 - m2).abs().max().item():.3g} "
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


@needs_cuda
@pytest.mark.parametrize("level", [8, 32])
def test_gradients_after_two_adam_steps(level):
    net = build("MDv", level).cuda().train()
    opt = torch.optim.Adam(net.parameters(), lr=6e-4, betas=(0.5, 0.999), weight_decay=0.0)
    crit = nn.L1Loss()
    x = inputs(batch=4, device="cuda")
    t = inputs(batch=4, seed=9, device="cuda")
    for step in (1, 2):
        net.zero_grad()
        crit(net(x), t).backward()
        nonzero = {k: p.grad is not None and p.grad.abs().sum().item() > 0 for k, p in net.mixer.named_parameters()}
        if step == 1:  # zero out_proj: only out_proj itself gets a gradient at the first step
            assert {k for k, v in nonzero.items() if v} == {"out_proj.weight"}, nonzero
        opt.step()
    assert all(nonzero.values()), [k for k, v in nonzero.items() if not v]


@needs_cuda
@pytest.mark.parametrize("level", [8, 32])
def test_weights_init_does_not_touch_block(level):
    DfpNet, _ = models.original_modules()
    net = build("MDv", level, seed=3)
    b = build("B", seed=3)
    for k, v in b.state_dict().items():
        assert torch.equal(v, net.state_dict()[k]), k
    torch.manual_seed(3)
    skel = DfpNet.TurbNetG(channelExponent=5)
    skel.apply(DfpNet.weights_init)
    fresh = blocks.make_block("MDv", 32 * models.LEVEL_LAYER[level][1], level)
    for k, v in fresh.state_dict().items():
        assert torch.equal(v, net.mixer.state_dict()[k]), k
    assert torch.count_nonzero(net.mixer.out_proj.weight) == 0
    touched = copy.deepcopy(fresh)
    touched.apply(DfpNet.weights_init)  # negative control: the conv2d would be re-initialised
    assert any(not torch.equal(v, touched.state_dict()[k]) for k, v in fresh.state_dict().items())


@needs_cuda
def test_param_counts():
    print()
    for level, c in ((8, 128), (32, 64)):
        net = build("MDv", level)
        n = models.block_params(net)
        e, r, ns = 2 * c, math.ceil(c / 16), 16
        expected = (2 * c                       # LayerNorm of the block
                    + c * 2 * e                 # in_proj
                    + e * 9 + e                 # depthwise conv2d 3x3 + bias
                    + 4 * (r + 2 * ns) * e      # x_proj per direction
                    + 4 * (e * r + e)           # dt_proj weight + bias per direction
                    + 4 * e * ns + 4 * e        # A_log, D per direction
                    + 2 * e                     # out_norm
                    + e * c)                    # out_proj
        md = models.block_params(build("MD", level))
        print(f"  MDv{level}: block params {n} (formula {expected}; MD{level} {md}, +{n - md}), "
              f"total {models.count_params(net)} ({models.count_params(net) - models.B_PARAMS:+d} vs B)")
        assert n == expected
        official = OfficialSS2D(c)
        assert n == sum(p.numel() for p in official.parameters()) + 2 * c
        assert {k: v.shape for k, v in official.state_dict().items()} == \
            {k: v.shape for k, v in net.mixer.state_dict().items() if not k.startswith("norm.")}


@needs_cuda
@pytest.mark.parametrize("level,channels", [(8, 128), (32, 64)])
def test_fidelity_vs_official_forward(level, channels):
    """Same weights in our block and in the transcription of the official SS2D; compare SS2D outputs and the
    gradients of the input. Also checks the official initialisation statistics against ours (A, D exact)."""
    torch.manual_seed(0)
    ours = randomise_out_proj(blocks.SS2DVMamba(channels, "cuda")).cuda()
    official = OfficialSS2D(channels).cuda()
    for k in ("A_logs", "Ds"):  # deterministic initialisation: identical
        assert torch.equal(getattr(official, k), getattr(ours, k)), k
    missing, unexpected = official.load_state_dict({k: v for k, v in ours.state_dict().items()
                                                   if not k.startswith("norm.")}, strict=True), None
    t = torch.randn(3, level, level, channels, device="cuda")
    t1, t2 = t.clone().requires_grad_(), t.clone().requires_grad_()
    y1, y2 = ours.ss2d(t1), official(t2)
    d = (y1 - y2).abs().max().item()
    print(f"\nMDv{level} ours vs official SS2D: max |diff| {d:.3g} (max |out| {y2.abs().max().item():.3g})")
    torch.testing.assert_close(y1, y2, rtol=1e-5, atol=1e-5)
    g = torch.randn_like(y1)
    (y1 * g).sum().backward()
    (y2 * g).sum().backward()
    torch.testing.assert_close(t1.grad, t2.grad, rtol=1e-5, atol=1e-5)
    # the full block is x + SS2D(LN(x)) as VSSBlock (drop path 0)
    x = torch.randn(2, channels, level, level, device="cuda")
    with torch.no_grad():
        yb = ours(x)
        yo = x + official(ours.norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
    torch.testing.assert_close(yb, yo, rtol=1e-5, atol=1e-5)
