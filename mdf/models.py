"""Model registry and factory. "dfpnet" / variant "B" is the unmodified original TurbNetG from train/DfpNet.py;
the other variants (build_from_cfg) add one global-mixing block from mdf/blocks.py.

The original modules in train/ are imported by appending train/ at the END of sys.path,
so they can never shadow other modules (train/ contains generic names such as
dataset.py and utils.py).
"""

import math
import sys

import torch

from mdf import blocks
from mdf.data import REPO_ROOT

ORIGINAL_DIR = REPO_ROOT / "train"


def _ensure_original_on_path():
    path = str(ORIGINAL_DIR)
    if path not in sys.path:
        sys.path.append(path)


def original_modules():
    """Return the original (DfpNet, utils) modules from train/, checking where they were loaded from."""
    _ensure_original_on_path()
    import DfpNet
    import utils

    for mod in (DfpNet, utils):
        if not mod.__file__ or not mod.__file__.startswith(str(ORIGINAL_DIR)):
            raise ImportError(f"{mod.__name__} resolved to {mod.__file__}, expected {ORIGINAL_DIR}")
    return DfpNet, utils


def compute_lr(epoch, epochs, min_lr, max_lr):
    """The original LR schedule (train/utils.py: computeLR)."""
    _, utils = original_modules()
    return utils.computeLR(epoch, epochs, min_lr, max_lr)


def step_lr(step, total_steps, ref_epochs, lr):
    """LR at optimiser step `step` (0-based) of a step-budget run: the original schedule evaluated at the virtual
    epoch floor(step * ref_epochs / total_steps), applied only below the base LR (as scripts/train.py). With
    total_steps = ref_epochs * batches_per_epoch this equals, step by step, the LR of an epoch-defined run."""
    lr_e = compute_lr(step * ref_epochs // total_steps, ref_epochs, lr * 0.1, lr)
    return lr_e if lr_e < lr else lr


def build_dfpnet(expo=5, dropout=0.0):
    DfpNet, _ = original_modules()
    net = DfpNet.TurbNetG(channelExponent=expo, dropout=dropout)
    net.apply(DfpNet.weights_init)
    return net


# name -> builder(expo) -> nn.Module with weights already initialised.
MODELS = {
    "dfpnet": build_dfpnet,
}


def build_model(name, expo=5):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    return MODELS[name](expo=expo)


def count_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


# --- Variants with a global-mixing block (docs/lab-notebook.md, 2026-10-12) ---

B_PARAMS = 2332035  # TurbNetG, 32 base channels
VARIANTS = ("B", "B-large", "CBAM", "SA", "MD")
TEST_VARIANTS = ("identity",)  # unit tests only
LEVELS = (64, 32, 16, 8)
# level (feature-map size) -> encoder layer after which the block sits, and its channel multiplier
LEVEL_LAYER = {64: ("layer1", 1), 32: ("layer2", 2), 16: ("layer2b", 2), 8: ("layer3", 4)}
B_LARGE_CHANNELS = 33


def channel_exponent(channels):
    """TurbNetG only takes an exponent (channels = int(2**e + 0.5)); return one that gives exactly `channels`."""
    e = math.log2(channels)
    if int(2 ** e + 0.5) != channels:
        raise ValueError(f"cannot represent {channels} channels with TurbNetG's exponent")
    return e


def make_turbnet_mixer():
    DfpNet, _ = original_modules()

    class TurbNetGMixer(DfpNet.TurbNetG):
        """TurbNetG with one block after the encoder layer of the chosen level.

        The block output replaces that layer's output, so it feeds both the next encoder layer and
        the skip connection. weights_init is applied to the skeleton BEFORE the block is created:
        the block keeps its own initialisation, and the skeleton consumes the RNG exactly as B does."""

        def __init__(self, block_kind, level, channels=32, mamba_impl="cuda", dropout=0.0):
            super().__init__(channelExponent=channel_exponent(channels), dropout=dropout)
            self.apply(DfpNet.weights_init)  # skeleton only: the block does not exist yet
            layer, mult = LEVEL_LAYER[level]
            self.level, self.mix_after = level, layer
            self.mixer = blocks.make_block(block_kind, channels * mult, level, mamba_impl)

        def forward(self, x):  # copied from TurbNetG.forward; the mixer is applied after `mix_after`
            m = self.mix_after
            out1 = self.layer1(x)
            if m == "layer1":
                out1 = self.mixer(out1)
            out2 = self.layer2(out1)
            if m == "layer2":
                out2 = self.mixer(out2)
            out2b = self.layer2b(out2)
            if m == "layer2b":
                out2b = self.mixer(out2b)
            out3 = self.layer3(out2b)
            if m == "layer3":
                out3 = self.mixer(out3)
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
            dout2 = self.dlayer2(dout2b_out2)
            dout2_out1 = torch.cat([dout2, out1], 1)
            dout1 = self.dlayer1(dout2_out1)
            return dout1

    return TurbNetGMixer


def normalize_cfg(cfg):
    """Fill defaults and check a model configuration {variant, level, channels, mamba_impl}."""
    variant = cfg.get("variant", "B")
    if variant not in VARIANTS + TEST_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; available {VARIANTS}")
    channels = cfg.get("channels")
    if variant == "B-large":
        if channels not in (None, B_LARGE_CHANNELS):
            raise ValueError(f"B-large has {B_LARGE_CHANNELS} base channels, got {channels}")
        channels = B_LARGE_CHANNELS
    channels = 32 if channels is None else int(channels)
    has_block = variant not in ("B", "B-large")
    level = cfg.get("level")
    level = (8 if level is None else int(level)) if has_block else None
    if has_block and level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level}")
    return {"variant": variant, "level": level, "channels": channels,
            "mamba_impl": (cfg.get("mamba_impl") or "cuda") if variant == "MD" else None}


def build_from_cfg(cfg):
    """The single model factory (training, evaluation, tests). Weights already initialised."""
    cfg = normalize_cfg(cfg)
    if cfg["variant"] in ("B", "B-large"):
        net = build_dfpnet(expo=channel_exponent(cfg["channels"]))
        if net.layer1.layer1_conv.out_channels != cfg["channels"]:
            raise RuntimeError("TurbNetG channel count mismatch")
        return net
    return make_turbnet_mixer()(cfg["variant"], cfg["level"], cfg["channels"], cfg["mamba_impl"] or "cuda")


def model_cfg_from_run_config(run_cfg):
    """Model configuration of a run from its config.json; runs from before the variants are B."""
    if run_cfg.get("variant"):
        return normalize_cfg({k: run_cfg.get(k) for k in ("variant", "level", "channels", "mamba_impl")})
    expo = run_cfg.get("args", {}).get("expo", 5)
    return normalize_cfg({"variant": "B", "channels": int(2 ** expo + 0.5)})


def block_params(net):
    mixer = getattr(net, "mixer", None)
    return 0 if mixer is None else count_params(mixer)
