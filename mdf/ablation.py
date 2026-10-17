"""Ablations of MDv64 (night 8, docs/lab-notebook.md 2026-10-16; docs/preregistration-night8.md).

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
"""

import torch
import torch.nn as nn

from mdf import blocks

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
