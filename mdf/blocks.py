"""Global-mixing blocks inserted into the DfpNet encoder (docs/lab-notebook.md, 2026-10-12).

All blocks map (B, C, H, W) -> (B, C, H, W).

  SelfAttention2D  y = x + Wo(Attn(LN(x) + PE)); fixed 2D sin/cos PE, 4 heads, SDPA; Wo zero-initialised
  SS2DMamba        y = x + sum_k unscan_k(Mamba(scan_k(LN(x)))); 4 scan orders, ONE shared Mamba module,
                   the 4 sequences stacked in the batch; Mamba out_proj zero-initialised
  SS2DVMamba       y = x + SS2D(LN(x)), SS2D as in the VMamba paper (MDv, docs/lab-notebook.md 2026-10-14)
  CBAM             Woo et al. 2018, channel then spatial attention, multiplicative, no residual

The blocks keep their own initialisation (PyTorch defaults, Mamba's internal init, zero output
projections); the original weights_init is never applied to them (see mdf/models.py).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MAMBA_IMPLS = ("cuda", "ref")


def sincos_pe_2d(channels, height, width):
    """Fixed 2D sinusoidal encoding, shape (H*W, C), row-major tokens.

    First C/2 channels encode the row index, last C/2 the column index; within each half,
    sin and cos at frequencies 1 / 10000^(2i / (C/2)) (Vaswani et al. 2017, applied per axis)."""
    if channels % 4:
        raise ValueError(f"channels must be divisible by 4, got {channels}")
    half = channels // 2
    freqs = 1.0 / (10000 ** (torch.arange(0, half, 2, dtype=torch.float64) / half))  # (half/2,)

    def axis(n):
        ang = torch.arange(n, dtype=torch.float64)[:, None] * freqs[None]  # (n, half/2)
        return torch.cat([ang.sin(), ang.cos()], dim=1)  # (n, half)

    rows = axis(height)[:, None, :].expand(height, width, half)
    cols = axis(width)[None, :, :].expand(height, width, half)
    return torch.cat([rows, cols], dim=2).reshape(height * width, channels).float()


class SelfAttention2D(nn.Module):
    def __init__(self, channels, height, width, heads=4):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels {channels} not divisible by heads {heads}")
        self.heads = heads
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.out = nn.Linear(channels, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # fixed, not trainable, not in the state_dict (recomputed from the shape)
        self.register_buffer("pe", sincos_pe_2d(channels, height, width), persistent=False)

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        t = self.norm(t) + self.pe
        q, k, v = self.qkv(t).view(b, h * w, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)  # (B, heads, HW, C/heads)
        a = a.transpose(1, 2).reshape(b, h * w, c)
        return x + self.out(a).transpose(1, 2).reshape(b, c, h, w)


# --- SS2D scan orders on channel-last maps (B, H, W, C) -> sequences (B, L, C) ---
SCAN_ORDERS = ("rows", "rows_rev", "cols", "cols_rev")


def scan(x, k):
    """x: (B, H, W, C) -> (B, H*W, C) in scan order k (0..3)."""
    b, h, w, c = x.shape
    s = x.reshape(b, h * w, c) if k in (0, 1) else x.transpose(1, 2).reshape(b, h * w, c)
    return s.flip(1) if k in (1, 3) else s


def unscan(s, k, h, w):
    """Inverse of scan: (B, H*W, C) -> (B, H, W, C)."""
    b, _, c = s.shape
    if k in (1, 3):
        s = s.flip(1)
    return s.reshape(b, h, w, c) if k in (0, 1) else s.reshape(b, w, h, c).transpose(1, 2)


class SS2DMamba(nn.Module):
    def __init__(self, channels, mamba_impl="cuda", d_state=16, d_conv=4, expand=2):
        super().__init__()
        if mamba_impl not in MAMBA_IMPLS:
            raise ValueError(f"mamba_impl must be one of {MAMBA_IMPLS}")
        from mamba_ssm import Mamba

        self.mamba_impl = mamba_impl
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        nn.init.zeros_(self.mamba.out_proj.weight)
        if self.mamba.out_proj.bias is not None:
            nn.init.zeros_(self.mamba.out_proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        t = self.norm(x.permute(0, 2, 3, 1))  # (B, H, W, C)
        seq = torch.cat([scan(t, k) for k in range(4)], dim=0).contiguous()  # (4B, HW, C)
        if self.mamba_impl == "cuda":
            out = self.mamba(seq)
        else:
            from mdf.mamba_ref import mamba_forward_ref
            out = mamba_forward_ref(self.mamba, seq)
        mixed = sum(unscan(o, k, h, w) for k, o in enumerate(out.chunk(4, dim=0)))
        return x + mixed.permute(0, 3, 1, 2)


class SS2DVMamba(nn.Module):
    """y = x + SS2D(LN(x)) with SS2D faithful to the VMamba paper version (Liu et al. 2024, arXiv 2401.10166).

    Followed: MzeroMiko/VMamba, commit 9ca24aee4bbffdb34619151afe8aa7a6ca278e44 (2024-01-19),
    classification/models/vmamba/vmamba.py, class SS2D (forward / forward_core), as used by VSSM (config
    TYPE vssm) through VSSBlock (x + drop_path(SS2D(ln_1(x)))). Re-implemented here (MIT-licensed upstream;
    the repository had no LICENSE file at that commit, it is MIT today): in_proj (C -> 2E, no bias) to x and z;
    depthwise 3x3 conv (bias) + SiLU on x; 4 scans (rows, columns, both reversed) with separate x_proj, dt_proj
    (weight, bias), A_log and D per direction; selective scan with grouped B / C (one group per direction);
    the 4 outputs mapped back to row-major order and summed; LayerNorm(E); * SiLU(z); out_proj (E -> C, no bias).
    E = expand * C, d_state 16, dt_rank ceil(C / 16); initialisation as upstream (dt "random", dt in
    [1e-3, 0.1], A = 1..N, D = 1; in_proj trunc_normal std 0.02 as VSSM._init_weights does for Linear layers).
    Declared deviations: out_proj zero-initialised (as MD / SA); no drop path; no dropout."""

    K = 4

    def __init__(self, channels, mamba_impl="cuda", d_state=16, d_conv=3, expand=2, dt_min=0.001, dt_max=0.1,
                 dt_init_floor=1e-4):
        super().__init__()
        if mamba_impl not in MAMBA_IMPLS:
            raise ValueError(f"mamba_impl must be one of {MAMBA_IMPLS}")
        self.mamba_impl = mamba_impl
        self.d_state = d_state
        self.d_inner = e = expand * channels
        self.dt_rank = r = math.ceil(channels / 16)
        self.norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, 2 * e, bias=False)
        nn.init.trunc_normal_(self.in_proj.weight, std=0.02)
        self.conv2d = nn.Conv2d(e, e, d_conv, padding=(d_conv - 1) // 2, groups=e, bias=True)
        # per direction: x_proj (R + 2N, E) with nn.Linear's default init, as upstream (4 Linear layers stacked)
        self.x_proj_weight = nn.Parameter(torch.stack([nn.Linear(e, r + 2 * d_state, bias=False).weight
                                                       for _ in range(self.K)]))
        dt_w, dt_b = [], []
        for _ in range(self.K):
            std = r ** -0.5
            dt_w.append(torch.empty(e, r).uniform_(-std, std))
            dt = torch.exp(torch.rand(e) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            dt = dt.clamp(min=dt_init_floor)
            dt_b.append(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.dt_projs_weight = nn.Parameter(torch.stack(dt_w))  # (K, E, R)
        self.dt_projs_bias = nn.Parameter(torch.stack(dt_b))  # (K, E)
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.K * e, 1)
        self.A_logs = nn.Parameter(torch.log(a))  # (K*E, N)
        self.Ds = nn.Parameter(torch.ones(self.K * e))  # (K*E,)
        self.out_norm = nn.LayerNorm(e)
        self.out_proj = nn.Linear(e, channels, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def _scan(self, *args, **kwargs):
        if self.mamba_impl == "cuda":
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            return selective_scan_fn(*args, **kwargs)
        from mdf.mamba_ref import selective_scan_ref
        return selective_scan_ref(*args, **kwargs)

    def forward_core(self, x):
        """x: (B, E, H, W) -> sum of the 4 directional outputs, (B, E, L) in row-major order."""
        b, e, h, w = x.shape
        l, k = h * w, self.K
        x_hwwh = torch.stack([x.reshape(b, e, l), x.transpose(2, 3).reshape(b, e, l)], dim=1)  # rows, columns
        xs = torch.cat([x_hwwh, x_hwwh.flip(-1)], dim=1)  # (B, K, E, L)
        x_dbl = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj_weight)
        dts, bs, cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight)
        out = self._scan(xs.float().reshape(b, k * e, l), dts.contiguous().float().reshape(b, k * e, l),
                         -torch.exp(self.A_logs.float()), bs.float().contiguous(), cs.float().contiguous(),
                         self.Ds.float(), z=None, delta_bias=self.dt_projs_bias.float().reshape(-1),
                         delta_softplus=True).reshape(b, k, e, l)
        inv = out[:, 2:4].flip(-1)
        wh = out[:, 1].reshape(b, e, w, h).transpose(2, 3).reshape(b, e, l)
        invwh = inv[:, 1].reshape(b, e, w, h).transpose(2, 3).reshape(b, e, l)
        return out[:, 0] + inv[:, 0] + wh + invwh

    def ss2d(self, t):
        """SS2D on channel-last input t: (B, H, W, C) -> (B, H, W, C)."""
        b, h, w, _ = t.shape
        xz = self.in_proj(t)
        xi, z = xz.chunk(2, dim=-1)
        xi = F.silu(self.conv2d(xi.permute(0, 3, 1, 2).contiguous()))
        y = self.forward_core(xi).transpose(1, 2).reshape(b, h, w, -1)
        return self.out_proj(self.out_norm(y) * F.silu(z))

    def forward(self, x):
        t = self.norm(x.permute(0, 2, 3, 1))
        return x + self.ss2d(t).permute(0, 3, 1, 2)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels))
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x.mean(dim=(2, 3))) + self.mlp(x.amax(dim=(2, 3))))
        x = x * ca[:, :, None, None]
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(dim=1, keepdim=True),
                                                   x.amax(dim=1, keepdim=True)], dim=1)))
        return x * sa


def make_block(kind, channels, size, mamba_impl="cuda"):
    if kind == "SA":
        return SelfAttention2D(channels, size, size)
    if kind == "MD":
        return SS2DMamba(channels, mamba_impl)
    if kind == "MDv":
        return SS2DVMamba(channels, mamba_impl)
    if kind == "MDv-local":  # ablation of MDv (night 8): no propagation along the scan
        from mdf.ablation import SS2DVMambaLocal
        return SS2DVMambaLocal(channels, mamba_impl)
    if kind == "Conv":  # plain residual conv block (night 8b, mdf/ablation.py)
        from mdf.ablation import ConvResBlock64
        return ConvResBlock64(channels)
    if kind.startswith("L-"):  # one-ingredient ablations of MDv-local (night 9, mdf/ablation.py)
        from mdf.ablation import SS2DVMambaLocalLadder
        return SS2DVMambaLocalLadder(channels, kind, mamba_impl)
    if kind == "CBAM":
        return CBAM(channels)
    if kind == "identity":
        return nn.Identity()
    raise ValueError(f"unknown block {kind!r}")

