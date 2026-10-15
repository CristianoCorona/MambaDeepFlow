"""Replacement of DfpNet's deep path with Mamba blocks (docs/lab-notebook.md, 2026-10-15, night 6b).

  TR     control: TurbNetG (32 base channels) truncated at 8x8. Kept unchanged: layer1, layer2, layer2b, layer3 and
         dlayer3, dlayer2b, dlayer2, dlayer1. Removed: layer4, layer5, layer6, dlayer6, dlayer5, dlayer4.
         dlayer3 receives cat([out3, out3]) (the same 256 input channels as cat([dout4, out3]) in the original).
  DRv11  as TR, with h = Stack(out3) at 8x8 / 128 channels and dlayer3(cat([h, out3])). Stack = 11 residual blocks
         x + SS2D(LN(x)) in sequence, each with its own weights: the MDv block of MDv8 (mdf/blocks.py SS2DVMamba,
         out_proj zero-initialised, no drop path). At initialisation every block is the identity, so DRv11 = TR.

Initialisation: weights_init is applied to the FULL TurbNetG skeleton before the deep layers are deleted and before
the blocks are created, so the RNG is consumed exactly as for B: with the same seed, the kept layers of TR / DRv11
start from the same weights as in B, and the blocks keep their own initialisation (as TurbNetGMixer in mdf/models.py).
"""

import torch

from mdf import blocks

DEEP_LAYERS = ("layer4", "layer5", "layer6", "dlayer6", "dlayer5", "dlayer4")
DR_LEVEL = 8  # feature-map size after layer3
N_BLOCKS = {"TR": 0, "DRv11": 11}  # N = 11 fixed in the pre-registration (docs/preregistration-night6b.md)


def make_turbnet_truncated():
    from mdf.models import original_modules

    DfpNet, _ = original_modules()

    class TurbNetGTruncated(DfpNet.TurbNetG):
        """TurbNetG without its deep path; optionally a stack of MDv blocks at 8x8 in its place (see module doc).

        The blocks are in `mixers` (nn.Sequential, indices 0..n-1), so models.block_params counts them."""

        def __init__(self, n_blocks, channels=32, mamba_impl="cuda", dropout=0.0):
            from mdf.models import channel_exponent

            super().__init__(channelExponent=channel_exponent(channels), dropout=dropout)
            self.apply(DfpNet.weights_init)  # full skeleton, as B: same RNG consumption
            for name in DEEP_LAYERS:
                delattr(self, name)
            self.n_blocks = n_blocks
            self.mixers = torch.nn.Sequential(*[blocks.make_block("MDv", channels * 4, DR_LEVEL, mamba_impl)
                                                for _ in range(n_blocks)])

        def forward(self, x):  # TurbNetG.forward with the deep path replaced by `mixers` (identity if empty)
            out1 = self.layer1(x)
            out2 = self.layer2(out1)
            out2b = self.layer2b(out2)
            out3 = self.layer3(out2b)
            h = self.mixers(out3)
            h_out3 = torch.cat([h, out3], 1)
            dout3 = self.dlayer3(h_out3)
            dout3_out2b = torch.cat([dout3, out2b], 1)
            dout2b = self.dlayer2b(dout3_out2b)
            dout2b_out2 = torch.cat([dout2b, out2], 1)
            dout2 = self.dlayer2(dout2b_out2)
            dout2_out1 = torch.cat([dout2, out1], 1)
            dout1 = self.dlayer1(dout2_out1)
            return dout1

    return TurbNetGTruncated


def build(variant, channels=32, mamba_impl="cuda"):
    return make_turbnet_truncated()(N_BLOCKS[variant], channels, mamba_impl or "cuda")
