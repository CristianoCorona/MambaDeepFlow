"""Ablations of MDv64: which part of the 64x64 Mamba block gives the gain? (poster, finding 3)

MDv64 lowers the v_y error, but it changes several things at once: a global scan, a gate, normalisations, a channel
expansion and a local 3x3 depthwise convolution. Each variant here removes or replaces ONE of them:

  MDv-local  (M64-loc)  the scan removed: no information travels along the sequence, every pixel only sees its 3x3
                        neighbourhood. Most of the gain stays, so the gain does NOT need the state space model.
                        This is the FINAL ARCHITECTURE (U-Net + M64-loc: +7 k parameters, 1.6x training time).
  MDv-dec    (M64-dec)  the unchanged MDv block on the decoder side at 64x64 instead of the encoder.
  Conv       (Conv64)   a plain residual conv block of a similar size (control: is any extra local layer enough?).
  L-*                   one-ingredient removals from MDv-local (gate, norms, expansion, depthwise -> dense conv).

How the scan is removed. In SS2DVMamba each direction computes y = SelectiveScan(u) + D * u, where the D term is a
per-channel skip that bypasses the state (see the Mamba equations in mdf/blocks.py). MDv-local keeps only D * u: in
row-major order this is D_k * x at the same pixel for every direction k, so the block becomes

      (xi, z) = in_proj(LN(x))
      y = x + out_proj( LN( (sum_k D_k) * SiLU(dwconv3x3(xi)) ) * SiLU(z) )

i.e. a gated, normalised, depthwise-convolutional block: a 3x3 receptive field and no long-range path.

Details of each variant (night 8, docs/lab-notebook.md 2026-10-16; docs/preregistration-night8.md):

Each variant differs from MDv64 (SS2DVMamba after layer1, 32 channels, 64x64) in one thing only:

  MDv-local  SS2DVMambaLocal: the selective-scan output of each of the 4 directions is replaced by its D-skip term
             D_k * x alone (the term the scan already adds), so there is no propagation along the scan; everything
             else as SS2DVMamba (in_proj, depthwise 3x3 conv + SiLU, sum of the 4 directions, out_norm, * SiLU(z),
             zero-initialised out_proj). Same __init__, hence the same RNG consumption and, at a given seed, the same
             weights as MDv; x_proj_weight, dt_projs_weight, dt_projs_bias and A_logs stay in the module (same state
             dict) but are not used and receive no gradient. The output at a pixel depends only on its 3x3
             neighbourhood (the depthwise conv).
  MDv-dec    TurbNetGDecMixer: the unchanged MDv block (32 channels, 64x64) on the decoder side, applied to the output
             of dlayer2 (32 channels, 64x64) before its concatenation [dout2, out1] that feeds dlayer1. The block is
             created after weights_init on the skeleton, as in TurbNetGMixer, so at a given seed its weights equal
             those of the MDv64 block.

SA64 (variant SA, level 64) and MDv16 (variant MDv, level 16) use the existing code unchanged.

Night 8b (docs/lab-notebook.md 2026-10-17; docs/preregistration-night8b.md), exploratory, after the night-8a results:

  Conv       ConvResBlock64: a plain residual convolutional block at the place of MDv64 (after layer1, 32 channels,
             64x64), y = x + Conv3x3(12 -> 32)(LeakyReLU(0.2)(Conv3x3(32 -> 12)(x))), padding 1, with bias; no
             normalisation, no gating. The last convolution (weight and bias) is zero-initialised, so the network equals
             B at initialisation; the first convolution keeps the PyTorch default initialisation (weights_init is not
             applied to blocks). 6956 parameters at 32 channels (MDv-local uses 7232). Receptive field of the block
             5x5 (two 3x3 convolutions; the depthwise conv of MDv-local gives 3x3).

Night 9 (docs/lab-notebook.md 2026-10-17; docs/preregistration-night9.md), exploratory, after nights 8a and 8b: a
ladder of one-ingredient ablations of MDv-local (SS2DVMambaLocalLadder, at the place of MDv64). Each differs from
MDv-local in ONE thing only; out_proj stays zero-initialised (network = B at init); the 3x3 receptive field is kept.

  L-nogate     no gating: the output is not multiplied by SiLU(z); in_proj still produces z (same weights as
               MDv-local) but z is not used, so the z rows of in_proj get a zero gradient.
  L-nonorm     out_norm (LayerNorm of the SS2D output) replaced by the identity.
  L-noprenorm  the LayerNorm at the input of the residual block replaced by the identity: y = x + SS2D(x).
  L-noexp      no channel expansion: inner channels E = 32 instead of 64 (expand = 1). Changes the parameter count and,
               since the shapes differ, the initial weights; it equals SS2DVMambaLocal(32, expand=1) at the same seed.
  L-dense      dense 3x3 convolution (E -> E, bias, PyTorch default initialisation) instead of the depthwise one.
               Changes the parameter count. Created after the parent __init__, so every other weight equals MDv-local's
               at the same seed.

L-nogate, L-nonorm, L-noprenorm: same __init__ as MDv-local (LayerNorm consumes no RNG), hence the same weights at a
given seed (minus the removed LayerNorm for L-nonorm / L-noprenorm).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mdf import blocks

# Scan parameters that MDv-local inherits from SS2DVMamba but never uses (kept so that the state dict and the
# random initialisation match MDv's; excluded from the "used" parameter count, 7232 instead of 20800 at 32 channels).
UNUSED_LOCAL = ("x_proj_weight", "dt_projs_weight", "dt_projs_bias", "A_logs")


class SS2DVMambaLocal(blocks.SS2DVMamba):
    """SS2DVMamba without long-range propagation: forward_core returns sum_k D_k * x (see the module docstring)."""

    def forward_core(self, x):
        """x: (B, E, H, W) -> (B, E, L) in row-major order. Each direction's scan output is replaced by D_k * u_k;
        mapped back to row-major order, D_k * u_k is D_k * x at the same pixel, so the 4 directions sum to
        (sum_k D_k) * x. Computed in float32 as the scan does."""
        b, e, h, w = x.shape
        d = self.Ds.float().view(self.K, e)  # (K, E), direction k = rows, columns, rows reversed, columns reversed
        xf = x.float().reshape(b, e, h * w)
        return sum(d[k][None, :, None] * xf for k in range(self.K))


class ConvResBlock64(nn.Module):
    """Residual block x + conv2(LeakyReLU(0.2)(conv1(x))), 3x3 convolutions with padding 1 and bias, conv2 zero."""

    def __init__(self, channels=32, hidden=12):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden, 3, padding=1, bias=True)
        self.act = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(hidden, channels, 3, padding=1, bias=True)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class SS2DVMambaLocalLadder(SS2DVMambaLocal):
    """MDv-local without one ingredient (`kind`, one of models.LADDER_VARIANTS; see the module docstring)."""

    def __init__(self, channels, kind, mamba_impl="cuda"):
        from mdf.models import LADDER_VARIANTS
        if kind not in LADDER_VARIANTS:
            raise ValueError(f"kind must be one of {LADDER_VARIANTS}, got {kind!r}")
        super().__init__(channels, mamba_impl, expand=1 if kind == "L-noexp" else 2)
        self.kind = kind
        if kind == "L-nonorm":
            self.out_norm = nn.Identity()
        if kind == "L-noprenorm":
            self.norm = nn.Identity()
        if kind == "L-dense":  # replaces the depthwise conv in place (same state-dict position)
            dw = self.conv2d
            self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, dw.kernel_size, padding=dw.padding, bias=True)

    def ss2d(self, t):
        if self.kind != "L-nogate":
            return super().ss2d(t)
        b, h, w, _ = t.shape
        xi, _ = self.in_proj(t).chunk(2, dim=-1)  # z computed as in MDv-local, not used
        xi = F.silu(self.conv2d(xi.permute(0, 3, 1, 2).contiguous()))
        y = self.forward_core(xi).transpose(1, 2).reshape(b, h, w, -1)
        return self.out_proj(self.out_norm(y))

    def unused_rows(self):
        """Parameters used only in part: {name: slice of unused rows} (the z rows of in_proj for L-nogate)."""
        e = self.d_inner
        return {"in_proj.weight": slice(e, 2 * e)} if self.kind == "L-nogate" else {}


def used_param_count(module):
    """Number of block parameters that the forward uses (unused parameters and unused rows excluded)."""
    unused = unused_param_names(module)
    rows = module.unused_rows() if isinstance(module, SS2DVMambaLocalLadder) else {}
    params = dict(module.named_parameters())
    return (sum(p.numel() for k, p in params.items() if k not in unused)
            - sum(params[k][sl].numel() for k, sl in rows.items()))


def unused_param_names(module):
    """Names of the parameters that the forward does not use (MDv-local only; empty otherwise)."""
    return set(UNUSED_LOCAL) if isinstance(module, SS2DVMambaLocal) else set()


def make_turbnet_dec_mixer():
    from mdf.models import channel_exponent, original_modules

    DfpNet, _ = original_modules()

    class TurbNetGDecMixer(DfpNet.TurbNetG):
        """TurbNetG with one MDv block on the decoder side at 64x64 (after dlayer2, before the concatenation with out1).

        As TurbNetGMixer: weights_init on the skeleton first, then the block (attribute `mixer`)."""

        def __init__(self, channels=32, mamba_impl="cuda", dropout=0.0):
            super().__init__(channelExponent=channel_exponent(channels), dropout=dropout)
            self.apply(DfpNet.weights_init)  # skeleton only: the block does not exist yet
            self.level, self.mix_after = 64, "dlayer2"
            self.mixer = blocks.make_block("MDv", channels, 64, mamba_impl)  # dlayer2 outputs `channels`

        def forward(self, x):  # copied from TurbNetG.forward; the mixer is applied to the output of dlayer2
            out1 = self.layer1(x)
            out2 = self.layer2(out1)
            out2b = self.layer2b(out2)
            out3 = self.layer3(out2b)
            out4 = self.layer4(out3)
            out5 = self.layer5(out4)
            out6 = self.layer6(out5)
            dout6 = self.dlayer6(out6)
            dout6_out5 = torch.cat([dout6, out5], 1)
            dout5 = self.dlayer5(dout6_out5)
            dout5_out4 = torch.cat([dout5, out4], 1)
            dout4 = self.dlayer4(dout5_out4)
            dout4_out3 = torch.cat([dout4, out3], 1)
            dout3 = self.dlayer3(dout4_out3)
            dout3_out2b = torch.cat([dout3, out2b], 1)
            dout2b = self.dlayer2b(dout3_out2b)
            dout2b_out2 = torch.cat([dout2b, out2], 1)
            dout2 = self.mixer(self.dlayer2(dout2b_out2))
            dout2_out1 = torch.cat([dout2, out1], 1)
            dout1 = self.dlayer1(dout2_out1)
            return dout1

    return TurbNetGDecMixer
