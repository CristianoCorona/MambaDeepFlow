"""Float32 data loading, fixed train/validation split and normalisation.

Re-implements the preprocessing of train/dataset.py (LoaderNormalizer with its default
switches fixedAirfoilNormalization = makeDimLess = removePOffset = True) without
float64 dataset-sized arrays: each .npz file (stored as float64) is cast to float32
as soon as it is read, and all normalisation steps run on float32 arrays.

Deviations from the original are declared in docs/lab-notebook.md (D1, D2, D8).
"""

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO_ROOT / "data" / "train"
TEST_DIR = REPO_ROOT / "data" / "test"
SPLITS_DIR = REPO_ROOT / "splits"

# Constant split seed (protocol decision, docs/lab-notebook.md). Never tie it to --seed.
SPLIT_SEED = 0
# Same rule as the original: min(20 %, 400) validation samples.
MAX_VAL = 400

# Fixed maxima from train/dataset.py (fixedAirfoilNormalization, makeDimLess).
MAX_INPUTS = (100.0, 38.12, 1.0)
MAX_TARGETS = (4.65, 2.04, 2.37)


def list_files(directory):
    """Sorted .npz file names, as the original (listdir + sort)."""
    return sorted(p.name for p in Path(directory).iterdir() if p.name.endswith(".npz"))


def split_files(files, split_seed=SPLIT_SEED):
    """Fixed split: shuffle the sorted list once with split_seed, last n_val files = validation.

    Mirrors the original (random.shuffle of the sorted list, then the last min(20 %, 400)
    samples become validation), but with a constant seed instead of the run seed.
    """
    files = sorted(files)
    random.Random(split_seed).shuffle(files)
    n_val = min(int(len(files) * 0.2), MAX_VAL)
    n_train = len(files) - n_val
    return files[:n_train], files[n_train:]


def profile(name):
    """Airfoil profile of a sample file: "<airfoil>_<fsX>_<fsY>.npz" -> "<airfoil>" (data/dataGen.py naming)."""
    airfoil, fsx, fsy = name[:-len(".npz")].rsplit("_", 2)
    int(fsx), int(fsy)  # both must be integers, else the naming assumption is wrong
    return airfoil


def split_files_grouped(files, split_seed=SPLIT_SEED):
    """Profile-disjoint split: whole profiles go to validation until it holds at least MAX_VAL files.

    The sorted list of unique profiles is shuffled once with split_seed; profiles are taken in that
    order until the validation set reaches MAX_VAL files (it may exceed it by part of one profile).
    No profile appears in both sets. Both lists are returned sorted by file name.
    """
    files = sorted(files)
    profiles = sorted({profile(f) for f in files})
    random.Random(split_seed).shuffle(profiles)
    by_profile = {}
    for f in files:
        by_profile.setdefault(profile(f), []).append(f)
    val_profiles, n_val = set(), 0
    for p in profiles:
        if n_val >= MAX_VAL:
            break
        val_profiles.add(p)
        n_val += len(by_profile[p])
    val = [f for f in files if profile(f) in val_profiles]
    train = [f for f in files if profile(f) not in val_profiles]
    return train, val


SPLIT_MODES = {"random": split_files, "grouped": split_files_grouped}


def split_file_path(split_mode, split_seed=SPLIT_SEED):
    """Committed validation list for a split mode (random keeps its historical name)."""
    name = f"val_seed{split_seed}.txt" if split_mode == "random" else f"val_grouped_seed{split_seed}.txt"
    return SPLITS_DIR / name


def file_list_sha256(files):
    """sha256 of the newline-joined file list (each name followed by '\\n')."""
    return hashlib.sha256("".join(f + "\n" for f in files).encode()).hexdigest()


def read_split_file(path):
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def load_raw(directory, files):
    """Read files into preallocated float32 arrays (inputs, targets), each (N, 3, 128, 128)."""
    n = len(files)
    inputs = np.empty((n, 3, 128, 128), dtype=np.float32)
    targets = np.empty((n, 3, 128, 128), dtype=np.float32)
    for i, name in enumerate(files):
        with np.load(Path(directory) / name) as npfile:
            d = npfile["a"]  # float64 (6, 128, 128); a single sample, cast immediately
        inputs[i] = d[0:3]
        targets[i] = d[3:6]
    return inputs, targets


def v_norm_of(inputs_raw_sample):
    """Free-stream speed |v| from the (un-normalised) input velocity channels, as the original."""
    return (np.max(np.abs(inputs_raw_sample[0])) ** 2 + np.max(np.abs(inputs_raw_sample[1])) ** 2) ** 0.5


def normalize_(inputs, targets):
    """In-place normalisation, identical in order and formulas to train/dataset.py.

    1. remove pressure offset: p -= mean(p); p -= p * mask
    2. dimensionless targets:  p /= |v|^2, vx /= |v|, vy /= |v|
    3. divide by fixed maxima (input channels 0 and 1, all target channels)
    Returns the per-sample |v| (float64) used in step 2.
    """
    n = inputs.shape[0]
    v_norms = np.empty(n, dtype=np.float64)
    for i in range(n):
        p = targets[i, 0]
        # float64 scalar accumulator for the mean (the original works in float64 throughout)
        p -= np.float32(np.mean(p, dtype=np.float64))
        p -= p * inputs[i, 2]
    for i in range(n):
        v_norm = v_norm_of(inputs[i])
        v_norms[i] = v_norm
        targets[i, 0] /= np.float32(v_norm ** 2)
        targets[i, 1] /= np.float32(v_norm)
        targets[i, 2] /= np.float32(v_norm)
    inputs[:, 0] *= np.float32(1.0 / MAX_INPUTS[0])
    inputs[:, 1] *= np.float32(1.0 / MAX_INPUTS[1])
    for c in range(3):
        targets[:, c] *= np.float32(1.0 / MAX_TARGETS[c])
    return v_norms


def denormalize(data, v_norm):
    """Inverse of the target normalisation for one sample (3, 128, 128); as TurbDataset.denormalize."""
    a = np.array(data, copy=True)
    for c in range(3):
        a[c] /= 1.0 / MAX_TARGETS[c]
    a[0] *= v_norm ** 2
    a[1] *= v_norm
    a[2] *= v_norm
    return a


def v_norm_from_normalized_input(inp):
    """|v| recomputed from a *normalised* input sample, exactly as runTrain.py / runTest.py do.

    Note: the original scripts compute v_norm from the already-normalised inputs (divided by
    100 and 38.12) before calling denormalize; reproduced here for the denormalised L1 metric.
    """
    return (np.max(np.abs(inp[0])) ** 2 + np.max(np.abs(inp[1])) ** 2) ** 0.5


class FlowDataset(Dataset):
    """In-memory normalised samples as float32 tensors."""

    def __init__(self, directory, files):
        self.files = list(files)
        inputs, targets = load_raw(directory, self.files)
        self.v_norms = normalize_(inputs, targets)
        self.inputs = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def load_train_val(train_dir=TRAIN_DIR, split_seed=SPLIT_SEED, split_file=None, split_mode="random"):
    """Return (train_ds, val_ds, split_info). If split_file is given, it must match the computed split."""
    all_files = list_files(train_dir)
    train_files, val_files = SPLIT_MODES[split_mode](all_files, split_seed)
    if split_file is not None:
        if not Path(split_file).exists():
            raise FileNotFoundError(f"Split file {split_file} not found")
        committed = read_split_file(split_file)
        if committed != val_files:
            raise RuntimeError(f"Computed validation split differs from {split_file}")
    info = {
        "split_mode": split_mode,
        "split_seed": split_seed,
        "n_files": len(all_files),
        "n_train": len(train_files),
        "n_val": len(val_files),
        "val_files_sha256": file_list_sha256(val_files),
        "train_files_sha256": file_list_sha256(train_files),
        "split_file": str(Path(split_file).relative_to(REPO_ROOT)) if split_file else None,
        "n_train_profiles": len({profile(f) for f in train_files}),
        "n_val_profiles": len({profile(f) for f in val_files}),
    }
    return FlowDataset(train_dir, train_files), FlowDataset(train_dir, val_files), info


def load_val(train_dir=TRAIN_DIR, split_seed=SPLIT_SEED, split_file=None, split_mode="random"):
    """Validation set only, checked against the committed split file (if given)."""
    _, val_files = SPLIT_MODES[split_mode](list_files(train_dir), split_seed)
    if split_file is not None and read_split_file(split_file) != val_files:
        raise RuntimeError(f"Computed validation split differs from {split_file}")
    return FlowDataset(train_dir, val_files)


def load_test(test_dir=TEST_DIR):
    return FlowDataset(test_dir, list_files(test_dir))
