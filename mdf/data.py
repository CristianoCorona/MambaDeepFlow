"""Float32 data loading, fixed train/validation split and normalisation.

The data (Thuerey et al. 2020; generated with OpenFOAM by data/dataGen.py). Each sample is one RANS simulation of
one airfoil at one free stream, resampled on a 128 x 128 grid around the airfoil and stored as a 6 x 128 x 128
array in "<airfoil>_<fsX>_<fsY>.npz":
  inputs   0  free-stream vx (constant in the fluid,      targets  3  pressure p
              0 inside the airfoil)                                4  velocity vx
           1  free-stream vy (same)                                5  velocity vy
           2  airfoil mask (1 inside the airfoil)                  (all targets 0 inside the airfoil)
The 6.4k set has 6400 training samples (1456 airfoil profiles, several free streams each) and a test set of 90
samples on 30 airfoils that never appear in training.

Normalisation (normalize_, identical in order and formulas to the original loader):
  1. remove the pressure offset (p minus its mean; 0 inside the airfoil);
  2. make the targets dimensionless with the free-stream speed |v|: p / |v|^2, vx / |v|, vy / |v|;
  3. divide by fixed maxima (MAX_INPUTS, MAX_TARGETS), so that every channel is of order 1.
The network learns this normalised field; errors are measured on it (scripts/evaluate.py).

Splits (split_files_for; the validation lists are committed in splits/ and checked on every load):
  random          the original rule (last min(20 %, 400) files of a shuffled list) with a constant seed. The same
                  airfoil then appears in training and validation, which flatters the validation error.
  grouped         profile-disjoint: whole airfoils go to validation (402 samples, 85 profiles). The default for
                  all reported runs: validation, like the test set, measures generalisation to unseen shapes.
  grouped-holdout grouped, plus a second validation of 400 seen shapes taken out of training (seen- vs unseen-shape
                  experiment, night 6 / 6b).
  kfold           profile-disjoint folds of data/train (prepared for cross-validation, not used in the results).
  grouped-full    the full 53.8k data set with the same 85 validation profiles (big-dataset experiment).
The split seed is a constant (SPLIT_SEED); it is never tied to the training seed, so all runs share one split.

Implementation: re-implements the preprocessing of train/dataset.py (LoaderNormalizer with its default switches
fixedAirfoilNormalization = makeDimLess = removePOffset = True) without float64 dataset-sized arrays: each .npz file
(stored as float64) is cast to float32 as soon as it is read, and all normalisation steps run on float32 arrays
(checked against the original by scripts/verify_loader.py).

Deviations from the original are declared in docs/lab-notebook.md (D1, D2, D8).
"""

import hashlib
import math
import os
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


def subsample_per_profile(files, fraction, seed):
    """Per-profile subset of training files (data-quantity experiment, docs/lab-notebook.md 2026-10-14).

    For a profile with n files, k = max(1, floor(fraction * n + 0.5)) files are kept (round half up): the first k of
    a permutation of its sorted files drawn with random.Random(f"{seed}:{profile}"). Every profile keeps >= 1 file;
    for the same seed a smaller fraction gives a subset of a larger one. Returns a sorted list."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    by_profile = {}
    for f in sorted(files):
        by_profile.setdefault(profile(f), []).append(f)
    keep = []
    for p, fs in sorted(by_profile.items()):
        perm = list(fs)
        random.Random(f"{seed}:{p}").shuffle(perm)
        keep += perm[:max(1, math.floor(fraction * len(fs) + 0.5))]
    return sorted(keep)


HOLDOUT = "grouped-holdout"
# seen-shape validation size of the grouped-holdout split (docs/lab-notebook.md, 2026-10-15 pre-registration)
N_SEEN = 400


def split_files_grouped_holdout_parts(files, split_seed=SPLIT_SEED):
    """Three-part split (docs/lab-notebook.md, 2026-10-15, "seen-shape vs unseen-shape validation"):
    (train, val_unseen, val_seen), all sorted.

    val_unseen = the grouped validation (split_files_grouped, unchanged). val_seen = N_SEEN files of the grouped
    training part: the sorted profiles with >= 2 files there are shuffled with random.Random(split_seed), the first
    N_SEEN are taken and from each one file is drawn with the same generator (rng.choice over its sorted files), so
    every seen profile keeps >= 1 file in training. train = grouped training part minus val_seen.
    """
    grouped_train, val_unseen = split_files_grouped(files, split_seed)
    by_profile = {}
    for f in grouped_train:  # sorted, so each profile's list is sorted
        by_profile.setdefault(profile(f), []).append(f)
    eligible = sorted(p for p, fs in by_profile.items() if len(fs) >= 2)
    if len(eligible) < N_SEEN:
        raise ValueError(f"only {len(eligible)} training profiles with >= 2 files, need {N_SEEN}")
    rng = random.Random(split_seed)
    rng.shuffle(eligible)
    val_seen = sorted(rng.choice(by_profile[p]) for p in eligible[:N_SEEN])
    seen = set(val_seen)
    return [f for f in grouped_train if f not in seen], val_unseen, val_seen


def split_files_grouped_holdout(files, split_seed=SPLIT_SEED):
    """(train, val_unseen) of the grouped-holdout split; val_unseen is the grouped validation."""
    train, val_unseen, _ = split_files_grouped_holdout_parts(files, split_seed)
    return train, val_unseen


SPLIT_MODES = {"random": split_files, "grouped": split_files_grouped, HOLDOUT: split_files_grouped_holdout}
KFOLD = "kfold"
ALL_SPLIT_MODES = (*SPLIT_MODES, KFOLD)


def kfold_profile_folds(files, n_folds, split_seed=SPLIT_SEED):
    """Profile-disjoint folds (docs/lab-notebook.md, 2026-10-13): list of n_folds sets of profiles.

    The sorted unique profiles are shuffled once with split_seed; each profile, in that order, goes to the fold
    that currently holds the fewest files (ties: lowest fold index), so folds are balanced in files. Every profile
    is in exactly one fold. Meant for the train + validation files (data/train), never for the test set.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    files = sorted(files)
    by_profile = {}
    for f in files:
        by_profile.setdefault(profile(f), []).append(f)
    profiles = sorted(by_profile)
    random.Random(split_seed).shuffle(profiles)
    folds, sizes = [set() for _ in range(n_folds)], [0] * n_folds
    for p in profiles:
        k = min(range(n_folds), key=lambda j: (sizes[j], j))
        folds[k].add(p)
        sizes[k] += len(by_profile[p])
    return folds


def split_files_kfold(files, n_folds, fold, split_seed=SPLIT_SEED):
    """(train, val) for one fold: val = files of the fold's profiles, train = all others; both sorted."""
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in 0..{n_folds - 1}, got {fold}")
    val_profiles = kfold_profile_folds(files, n_folds, split_seed)[fold]
    files = sorted(files)
    return ([f for f in files if profile(f) not in val_profiles], [f for f in files if profile(f) in val_profiles])


def split_files_for(split_mode, files, split_seed=SPLIT_SEED, n_folds=None, fold=None):
    """Dispatch over all split modes (kfold needs n_folds and fold)."""
    if split_mode == KFOLD:
        if n_folds is None or fold is None:
            raise ValueError("split mode kfold needs n_folds and fold")
        return split_files_kfold(files, n_folds, fold, split_seed)
    return SPLIT_MODES[split_mode](files, split_seed)


def split_file_path(split_mode, split_seed=SPLIT_SEED, n_folds=None, fold=None):
    """Committed validation list for a split mode (random keeps its historical name)."""
    if split_mode == KFOLD:
        return SPLITS_DIR / f"kfold{n_folds}_seed{split_seed}_fold{fold}.txt"
    # grouped-holdout: its unseen-shape validation is the grouped one, checked against the same file
    name = f"val_seed{split_seed}.txt" if split_mode == "random" else f"val_grouped_seed{split_seed}.txt"
    return SPLITS_DIR / name


def seen_split_file_path(split_seed=SPLIT_SEED):
    """Committed seen-shape validation list of the grouped-holdout split."""
    return SPLITS_DIR / f"val_seen_holdout_seed{split_seed}.txt"


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


def load_train_val(train_dir=TRAIN_DIR, split_seed=SPLIT_SEED, split_file=None, split_mode="random",
                   n_folds=None, fold=None, train_fraction=None, subset_seed=None):
    """Return (train_ds, val_ds, split_info). If split_file is given, it must match the computed split.
    train_fraction (with subset_seed): keep a per-profile subset of the training files (subsample_per_profile)."""
    all_files = list_files(train_dir)
    train_files, val_files = split_files_for(split_mode, all_files, split_seed, n_folds, fold)
    n_train_full = len(train_files)
    if train_fraction is not None:
        if subset_seed is None:
            raise ValueError("train_fraction needs subset_seed")
        train_files = subsample_per_profile(train_files, train_fraction, subset_seed)
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
    if split_mode == KFOLD:
        info.update({"n_folds": n_folds, "fold": fold})
    if split_mode == HOLDOUT:
        _, _, val_seen = split_files_grouped_holdout_parts(all_files, split_seed)
        if set(val_seen) & set(train_files) or set(val_seen) & set(val_files):
            raise RuntimeError("seen-shape validation overlaps train or unseen validation")
        seen_file = seen_split_file_path(split_seed)
        if split_file is not None and read_split_file(seen_file) != val_seen:
            raise RuntimeError(f"Computed seen-shape validation differs from {seen_file}")
        info.update({"n_val_seen": len(val_seen), "val_seen_files_sha256": file_list_sha256(val_seen),
                     "val_seen_split_file": str(seen_file.relative_to(REPO_ROOT)),
                     "n_val_seen_profiles": len({profile(f) for f in val_seen}),
                     "val_seen_rule": "one file from each of N_SEEN grouped-training profiles with >= 2 files, "
                                      "random.Random(split_seed) shuffle of the sorted profiles, then rng.choice"})
    if train_fraction is not None:
        info.update({"train_fraction": train_fraction, "subset_seed": subset_seed, "n_train_full": n_train_full,
                     "subset_rule": "per profile k = max(1, floor(f*n + 0.5)), first k of random.Random("
                                    "f'{seed}:{profile}') permutation of the sorted files"})
    return FlowDataset(train_dir, train_files), FlowDataset(train_dir, val_files), info


def load_val(train_dir=TRAIN_DIR, split_seed=SPLIT_SEED, split_file=None, split_mode="random",
             n_folds=None, fold=None):
    """Validation set only, checked against the committed split file (if given)."""
    _, val_files = split_files_for(split_mode, list_files(train_dir), split_seed, n_folds, fold)
    if split_file is not None and read_split_file(split_file) != val_files:
        raise RuntimeError(f"Computed validation split differs from {split_file}")
    return FlowDataset(train_dir, val_files)


def load_val_seen(train_dir=TRAIN_DIR, split_seed=SPLIT_SEED, split_file=None):
    """Seen-shape validation of the grouped-holdout split, checked against the committed list (if given)."""
    _, _, val_seen = split_files_grouped_holdout_parts(list_files(train_dir), split_seed)
    if split_file is not None and read_split_file(split_file) != val_seen:
        raise RuntimeError(f"Computed seen-shape validation differs from {split_file}")
    return FlowDataset(train_dir, val_seen)


def load_test(test_dir=TEST_DIR):
    return FlowDataset(test_dir, list_files(test_dir))


# --- Full dataset, split mode "grouped-full" (docs/lab-notebook.md, 2026-10-19; docs/preregistration-bigdata.md) ---
# data_full.tar.gz (README of thunil/Deep-Flow-Prediction) unpacks to data/test/ (the same 90 files as the 6.4k set)
# and data/train/<subdir>/<name>.npz with subdirs reg/ (regular airfoils, dataGen.py naming), shear/ (sheared airfoils:
# shearAirfoils.py appends "_<int(shear*1000)>" to the airfoil name) and sup/ (superimposed; naming unknown, not used).
# data/train_full is a symlink to the unpacked data/train/ (only on the cloud VM); the test set is never unpacked there.
FULL_DIR = REPO_ROOT / "data" / "train_full"
GROUPED_FULL = "grouped-full"
FULL_SUBDIRS_DEFAULT = ("reg", "shear")
FULL_SUBDIRS_KNOWN = ("reg", "shear", "sup")
TEST_PROFILES_FILE = SPLITS_DIR / "test_profiles.txt"


def full_subdirs(value=None):
    """Subdirectories of the full set used for training / validation: value, else $MDF_FULL_SUBDIRS, else reg,shear.

    The environment variable is the declared fallback of the pre-registration (e.g. "reg" if the shear names do not
    parse); the value used is stored in config.json and reused by evaluate.py."""
    text = value if value is not None else os.environ.get("MDF_FULL_SUBDIRS", ",".join(FULL_SUBDIRS_DEFAULT))
    subs = tuple(s for s in text.split(",") if s)
    if not subs or any(s not in FULL_SUBDIRS_KNOWN for s in subs) or len(set(subs)) != len(subs):
        raise ValueError(f"bad full-set subdirs {text!r}; known: {FULL_SUBDIRS_KNOWN}")
    return subs


def list_files_full(directory, subdirs):
    """Sorted relative paths "<subdir>/<name>.npz" of the given subdirectories (a missing subdirectory counts as
    empty). Other entries of the directory are ignored here; scripts/cloud/inspect_full.py reports them."""
    out = []
    for sub in subdirs:
        d = Path(directory) / sub
        if d.is_dir():
            out += [f"{sub}/{n}" for n in list_files(d)]
    return sorted(out)


def profile_full(relpath):
    """Base airfoil profile of a full-set file "<subdir>/<name>.npz".

    reg:   "<airfoil>_<fsX>_<fsY>.npz"          -> <airfoil>  (as profile(); the airfoil may contain underscores)
    shear: "<airfoil>_<shear>_<fsX>_<fsY>.npz"  -> <airfoil>  (the sheared shape of <airfoil>, same group)
    The subdirectory decides the rule, so a regular airfoil whose own name ends in "_<int>" (e.g. "hybrid_1") is not
    mistaken for a sheared one. Any other subdirectory or a name that does not parse raises ValueError."""
    sub, _, name = relpath.partition("/")
    if not name.endswith(".npz") or "/" in name:
        raise ValueError(f"unexpected full-set path {relpath!r}")
    if sub == "reg":
        try:
            return profile(name)
        except ValueError:
            raise ValueError(f"reg name does not parse: {relpath!r}") from None
    if sub == "shear":
        parts = name[:-len(".npz")].rsplit("_", 3)
        try:
            if len(parts) != 4 or not parts[0]:
                raise ValueError
            int(parts[1]), int(parts[2]), int(parts[3])
        except ValueError:
            raise ValueError(f"shear name does not parse: {relpath!r}") from None
        return parts[0]
    raise ValueError(f"no profile rule for subdirectory {sub!r} ({relpath!r})")


def grouped_val_profiles(split_seed=SPLIT_SEED):
    """Profiles of the committed grouped validation (splits/val_grouped_seed0.txt): 85 profiles."""
    return sorted({profile(f) for f in read_split_file(split_file_path("grouped", split_seed))})


def profiles_of_test():
    """Profiles of the 30 test airfoils (splits/test_profiles.txt, from the test file names only)."""
    return read_split_file(TEST_PROFILES_FILE)


def split_files_grouped_full(files, val_profiles, test_profiles_):
    """(train, val, excluded) of the full set, all sorted. val = every file whose profile is a validation profile;
    excluded = every file whose profile is a test profile (never used); train = all others."""
    val_p, test_p = set(val_profiles), set(test_profiles_)
    if val_p & test_p:
        raise ValueError(f"validation and test profiles overlap: {sorted(val_p & test_p)}")
    train, val, excluded = [], [], []
    for f in sorted(files):
        p = profile_full(f)
        (val if p in val_p else excluded if p in test_p else train).append(f)
    return train, val, excluded


def load_train_val_full(full_dir=FULL_DIR, subdirs=None, split_seed=SPLIT_SEED, train_fraction=None,
                        subset_seed=None):
    """(train_ds, val_ds, split_info) of split mode grouped-full. train_fraction / subset_seed as load_train_val (per
    base profile; only for the short cloud check, scripts/cloud/smoke_test.sh)."""
    subs = full_subdirs(None if subdirs is None else ",".join(subdirs))
    all_files = list_files_full(full_dir, subs)
    if not all_files:
        raise FileNotFoundError(f"no .npz files under {full_dir}/{{{','.join(subs)}}}")
    val_p, test_p = grouped_val_profiles(split_seed), profiles_of_test()
    train_files, val_files, excluded = split_files_grouped_full(all_files, val_p, test_p)
    n_train_full = len(train_files)
    if train_fraction is not None:
        if subset_seed is None:
            raise ValueError("train_fraction needs subset_seed")
        by_profile = {}
        for f in train_files:
            by_profile.setdefault(profile_full(f), []).append(f)
        keep = []
        for p, fs in sorted(by_profile.items()):
            perm = list(fs)
            random.Random(f"{subset_seed}:{p}").shuffle(perm)
            keep += perm[:max(1, math.floor(train_fraction * len(fs) + 0.5))]
        train_files = sorted(keep)
    found_val_p = {profile_full(f) for f in val_files}
    info = {
        "split_mode": GROUPED_FULL,
        "split_seed": split_seed,
        "full_dir": str(full_dir),
        "full_subdirs": list(subs),
        "n_files": len(all_files),
        "n_files_per_subdir": {s: sum(f.startswith(s + "/") for f in all_files) for s in subs},
        "n_train": len(train_files),
        "n_val": len(val_files),
        "n_excluded_test_profiles": len(excluded),
        "val_files_sha256": file_list_sha256(val_files),
        "train_files_sha256": file_list_sha256(train_files),
        "split_file": None,
        "val_profiles_source": str(split_file_path("grouped", split_seed).relative_to(REPO_ROOT)),
        "test_profiles_source": str(TEST_PROFILES_FILE.relative_to(REPO_ROOT)),
        "n_train_profiles": len({profile_full(f) for f in train_files}),
        "n_val_profiles": len(found_val_p),
        "val_profiles_missing": sorted(set(val_p) - found_val_p),
    }
    if train_fraction is not None:
        info.update({"train_fraction": train_fraction, "subset_seed": subset_seed, "n_train_full": n_train_full,
                     "subset_rule": "per base profile k = max(1, floor(f*n + 0.5)), first k of random.Random("
                                    "f'{seed}:{profile}') permutation of the sorted files"})
    return FlowDataset(full_dir, train_files), FlowDataset(full_dir, val_files), info


def load_val_full(full_dir=FULL_DIR, subdirs=None, split_seed=SPLIT_SEED):
    """Validation set of grouped-full (all files of the 85 grouped validation profiles in the used subdirs)."""
    subs = full_subdirs(None if subdirs is None else ",".join(subdirs))
    _, val_files, _ = split_files_grouped_full(list_files_full(full_dir, subs), grouped_val_profiles(split_seed),
                                               profiles_of_test())
    return FlowDataset(full_dir, val_files)
