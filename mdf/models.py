"""Model registry. "dfpnet" is the unmodified original TurbNetG from train/DfpNet.py.

The original modules in train/ are imported by appending train/ at the END of sys.path,
so they can never shadow other modules (train/ contains generic names such as
dataset.py and utils.py).
"""

import sys

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
