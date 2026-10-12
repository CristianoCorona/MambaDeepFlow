"""Global-mixing blocks inserted into the DfpNet encoder (docs/lab-notebook.md, 2026-10-12).

All blocks map (B, C, H, W) -> (B, C, H, W).

  SelfAttention2D  y = x + Wo(Attn(LN(x) + PE)); fixed 2D sin/cos PE, 4 heads, SDPA; Wo zero-initialised
  SS2DMamba        y = x + sum_k unscan_k(Mamba(scan_k(LN(x)))); 4 scan orders, ONE shared Mamba module,
                   the 4 sequences stacked in the batch; Mamba out_proj zero-initialised
  CBAM             Woo et al. 2018, channel then spatial attention, multiplicative, no residual

The blocks keep their own initialisation (PyTorch defaults, Mamba's internal init, zero output
projections); the original weights_init is never applied to them (see mdf/models.py).
"""

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
    if kind == "CBAM":
        return CBAM(channels)
    if kind == "identity":
        return nn.Identity()
    raise ValueError(f"unknown block {kind!r}")

