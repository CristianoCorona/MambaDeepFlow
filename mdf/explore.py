"""Exploratory per-sample / per-region analysis helpers (docs/lab-notebook.md, 2026-10-13).

Reusable for any split, but the test split is final-results-only: `load_split("test", ...)` refuses unless
allow_test=True, and every loader path goes through `assert_files_allowed`, which asserts that no file of
data/test is read for split "val".

Geometry conventions of the 128x128 arrays (checked on the data, notebook 2026-10-13): array axis 0 (rows) is
the streamwise x direction (the chord lies along it, the flow goes towards increasing row index), axis 1
(columns) is y; the free-stream direction in (row, col) index space is (vx_inf, vy_inf).

Regions (disjoint, in this priority order; all masks are (128, 128) bool):
  interior  airfoil cells (input channel 2 == 1); targets are 0 there
  wall      fluid cells at Euclidean distance 1..WALL_PX px (cell centres) from the nearest airfoil cell
            (proxy of the boundary layer)
  wake      remaining fluid cells downstream of the trailing edge along the free-stream direction:
            s = (cell - TE) . u > 0 and |(cell - TE) x u| <= WAKE_HALF_WIDTH px, u = unit free-stream vector,
            TE = last airfoil row (most downstream in x), mean column of its airfoil cells
  rest      all other fluid cells
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mdf import data as mdata
from mdf import models

WALL_PX = 4
WAKE_HALF_WIDTH = 5
REGIONS = ("interior", "wall", "wake", "rest")
CHANNEL_NAMES = ("p", "vx", "vy")
CACHE_VERSION = 2


# --- test-set guard ---

def assert_files_allowed(directory, files, split):
    """Explicit guard: for split 'val' every file must be read from data/train, never from data/test."""
    directory = Path(directory).resolve()
    test_dir = mdata.TEST_DIR.resolve()
    train_dir = mdata.TRAIN_DIR.resolve()
    if split == "val":
        assert directory == train_dir, f"val files must come from {train_dir}, got {directory}"
        assert directory != test_dir, "refusing to read the test directory for split 'val'"
        for f in files:
            path = (directory / f).resolve()
            assert test_dir not in path.parents, f"{path} is inside the test directory"
            assert path.parent == train_dir, f"{path} is not in {train_dir}"
    elif split != "test":
        raise ValueError(f"unknown split {split!r}")


def load_split(split, split_mode="grouped", allow_test=False):
    """FlowDataset for 'val' (profile-disjoint by default, checked against the committed split file) or 'test'."""
    if split == "test":
        if not allow_test:
            raise SystemExit("split 'test' is for final results only; pass --allow-test explicitly")
        files = mdata.list_files(mdata.TEST_DIR)
        assert_files_allowed(mdata.TEST_DIR, files, "test")
        return mdata.FlowDataset(mdata.TEST_DIR, files)
    _, val_files = mdata.SPLIT_MODES[split_mode](mdata.list_files(mdata.TRAIN_DIR))
    if mdata.read_split_file(mdata.split_file_path(split_mode)) != val_files:
        raise RuntimeError("computed validation split differs from the committed split file")
    assert_files_allowed(mdata.TRAIN_DIR, val_files, "val")
    return mdata.FlowDataset(mdata.TRAIN_DIR, val_files)


# --- per-sample geometry and properties (model independent) ---

def free_stream(inp_norm, mask):
    """(vx_inf, vy_inf) in the original units from a normalised input sample (fluid-cell median)."""
    fluid = ~mask
    vx = float(np.median(inp_norm[0][fluid])) * mdata.MAX_INPUTS[0]
    vy = float(np.median(inp_norm[1][fluid])) * mdata.MAX_INPUTS[1]
    return vx, vy


def distance_to_airfoil(mask):
    """Euclidean distance (px, cell centres) from every cell to the nearest airfoil cell (0 inside)."""
    m = mask.astype(bool)
    # boundary airfoil cells (with a 4-neighbour outside) are enough for the nearest-cell distance from outside
    pad = np.pad(m, 1, constant_values=False)
    inner = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    br, bc = np.nonzero(m & ~inner)
    rr, cc = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    d2 = (rr[..., None] - br) ** 2 + (cc[..., None] - bc) ** 2
    dist = np.sqrt(d2.min(axis=-1).astype(np.float64))
    dist[m] = 0.0
    return dist


def trailing_edge(mask):
    rows = np.nonzero(mask.any(axis=1))[0]
    te_row = rows[-1]
    return float(te_row), float(np.nonzero(mask[te_row])[0].mean())


def region_masks(mask, vx_inf, vy_inf):
    """Disjoint region masks (see the module docstring)."""
    m = mask.astype(bool)
    dist = distance_to_airfoil(m)
    wall = (~m) & (dist >= 1) & (dist <= WALL_PX)
    te_r, te_c = trailing_edge(m)
    u = np.array([vx_inf, vy_inf], dtype=np.float64)
    u /= np.linalg.norm(u)
    rr, cc = np.mgrid[0:m.shape[0], 0:m.shape[1]].astype(np.float64)
    dr, dc = rr - te_r, cc - te_c
    along = dr * u[0] + dc * u[1]
    across = np.abs(dr * u[1] - dc * u[0])
    wake = (~m) & (~wall) & (along > 0) & (across <= WAKE_HALF_WIDTH)
    rest = (~m) & (~wall) & (~wake)
    return {"interior": m, "wall": wall, "wake": wake, "rest": rest}


def thickness_ratio(mask):
    """Relative thickness from the mask: max over rows (x stations) of the airfoil cells in that row (y extent),
    divided by the chord in px (number of rows containing airfoil cells). Pixel-quantised (chord ~60 px)."""
    m = mask.astype(bool)
    per_row = m.sum(axis=1)
    chord = int((per_row > 0).sum())
    return float(per_row.max()) / chord, chord


def sample_properties(ds):
    """Model-independent per-sample table: file, profile, free stream, angle, pressure amplitude, thickness."""
    rows = []
    for i, f in enumerate(ds.files):
        inp = ds.inputs[i].numpy()
        tgt = ds.targets[i].numpy()
        mask = inp[2] > 0.5
        vx, vy = free_stream(inp, mask)
        thick, chord = thickness_ratio(mask)
        rows.append({
            "index": i, "file": f, "profile": mdata.profile(f),
            "vx_inf": vx, "vy_inf": vy,
            "alpha_deg": float(np.degrees(np.arctan2(vy, vx))),
            "abs_alpha_deg": float(abs(np.degrees(np.arctan2(vy, vx)))),
            "sum_abs_p_tgt": float(np.abs(tgt[0].astype(np.float64)).sum()),
            "thickness": thick, "chord_px": chord,
        })
    return rows


def all_region_masks(ds, props):
    return [region_masks(ds.inputs[i].numpy()[2] > 0.5, props[i]["vx_inf"], props[i]["vy_inf"])
            for i in range(len(ds))]


# --- inference with per-sample aggregates ---

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_inference(run_dir, ds, regions, keep_idx, device="cuda", batch_size=1):
    """Per-sample aggregates of one run (nothing per-pixel is kept except the mean |error| map and the
    predictions of the samples in keep_idx).

    Returns a dict of numpy arrays:
      rel_<c>, abs_err_<c>, abs_tgt_<c>    whole grid, c in p, vx, vy, all (float64 sums, as evaluate.py)
      reg_err_<r>   (N, 3) sum |o - t| per channel in region r;   reg_tgt_<r> (N, 3) sum |t|;  reg_n_<r> (N,)
      mean_abs_err_map  (3, 128, 128) mean over samples of |o - t|
      keep_pred (K, 3, 128, 128) float32, keep_idx (K,)
    """
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    mcfg = models.model_cfg_from_run_config(cfg)
    if mcfg["variant"] == "MD" and device == "cpu":
        mcfg["mamba_impl"] = "ref"  # CUDA kernels need the GPU; reference path, same parameters
    net = models.build_from_cfg(mcfg)
    net.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    net.to(device).eval()

    n = len(ds)
    out = {f"{k}_{c}": np.zeros(n) for k in ("rel", "abs_err", "abs_tgt") for c in (*CHANNEL_NAMES, "all")}
    for r in REGIONS:
        out[f"reg_err_{r}"] = np.zeros((n, 3))
        out[f"reg_tgt_{r}"] = np.zeros((n, 3))
        out[f"reg_n_{r}"] = np.zeros(n)
    err_map = np.zeros((3, 128, 128))
    keep_pos = {int(k): j for j, k in enumerate(keep_idx)}
    keep_pred = np.zeros((len(keep_idx), 3, 128, 128), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            x = ds.inputs[start:start + batch_size].to(device)
            pred = net(x).cpu().numpy()
            for j in range(pred.shape[0]):
                i = start + j
                o = pred[j].astype(np.float64)
                t = ds.targets[i].numpy().astype(np.float64)
                ae, at = np.abs(o - t), np.abs(t)
                err_map += ae
                for ci, c in enumerate(CHANNEL_NAMES):
                    out[f"abs_err_{c}"][i] = ae[ci].sum()
                    out[f"abs_tgt_{c}"][i] = at[ci].sum()
                out["abs_err_all"][i] = ae.sum()
                out["abs_tgt_all"][i] = at.sum()
                for r in REGIONS:
                    m = regions[i][r]
                    out[f"reg_err_{r}"][i] = ae[:, m].sum(axis=1)
                    out[f"reg_tgt_{r}"][i] = at[:, m].sum(axis=1)
                    out[f"reg_n_{r}"][i] = m.sum()
                if i in keep_pos:
                    keep_pred[keep_pos[i]] = pred[j]
    for c in (*CHANNEL_NAMES, "all"):
        out[f"rel_{c}"] = out[f"abs_err_{c}"] / out[f"abs_tgt_{c}"]
    out["mean_abs_err_map"] = err_map / n
    out["keep_pred"] = keep_pred
    out["keep_idx"] = np.asarray(keep_idx, dtype=np.int64)
    return out


def cached_inference(run_dir, ds, regions, keep_idx, cache_dir, split, device="cuda"):
    """run_inference with an on-disk cache keyed by checkpoint sha256, file list, keep list, region settings."""
    run_dir = Path(run_dir).resolve()
    key = {
        "version": CACHE_VERSION, "split": split, "model_sha256": file_sha256(run_dir / "model.pt"),
        "files_sha256": mdata.file_list_sha256(ds.files), "keep_idx": [int(k) for k in keep_idx],
        "wall_px": WALL_PX, "wake_half_width": WAKE_HALF_WIDTH, "batch_size": 1,
    }
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{run_dir.parent.name}__{run_dir.name}__{split}.npz"
    if path.exists():
        with np.load(path) as z:
            if json.loads(str(z["key"])) == key:
                return {k: z[k] for k in z.files if k != "key"}, True
    res = run_inference(run_dir, ds, regions, keep_idx, device)
    np.savez_compressed(path, key=json.dumps(key), **res)
    return res, False


# --- bootstrap over profiles ---

def profile_onehot(profiles):
    names = sorted(set(profiles))
    pos = {p: j for j, p in enumerate(names)}
    g = np.zeros((len(profiles), len(names)))
    for i, p in enumerate(profiles):
        g[i, pos[p]] = 1.0
    return g, names


def _ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def paired_bootstrap(m_var, m_ref, profiles, n_boot=10000, seed=0, two_level=True, subset=None):
    """Paired comparison var - ref of a per-sample metric (lower is better).

    m_var, m_ref: (seeds, N) arrays of the metric per seed and sample. subset: optional bool (N,) mask.
    One level: per-sample seed means, difference per sample, profiles resampled with replacement (mean over the
    samples of the resampled profiles, i.e. sample-weighted as the metric itself).
    Two levels: at each repetition the seeds of each configuration are also resampled with replacement.
    Returns point estimate, CIs, median per-sample difference, fraction of profiles / samples where var < ref.
    """
    m_var, m_ref = np.atleast_2d(m_var), np.atleast_2d(m_ref)
    if subset is not None:
        m_var, m_ref = m_var[:, subset], m_ref[:, subset]
        profiles = [p for p, s in zip(profiles, subset) if s]
    g, _ = profile_onehot(profiles)
    n_prof = g.shape[1]
    counts = g.sum(axis=0)
    d = m_var.mean(axis=0) - m_ref.mean(axis=0)
    prof_sum = d @ g
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_prof, size=(n_boot, n_prof))
    denom = counts[idx].sum(axis=1)
    boot1 = prof_sum[idx].sum(axis=1) / denom
    res = {
        "n_samples": int(d.size), "n_profiles": int(n_prof),
        "seeds_var": int(m_var.shape[0]), "seeds_ref": int(m_ref.shape[0]),
        "mean_diff": float(d.mean()), "ci1_lo": _ci(boot1)[0], "ci1_hi": _ci(boot1)[1],
        "median_diff": float(np.median(d)),
        "frac_profiles_var_better": float(np.mean(prof_sum / counts < 0)),
        "frac_samples_var_better": float(np.mean(d < 0)),
    }
    if two_level:
        def seed_resampled_means(m):
            k = m.shape[0]
            pick = rng.integers(0, k, size=(n_boot, k))
            w = np.zeros((n_boot, k))
            np.add.at(w, (np.arange(n_boot)[:, None], pick), 1.0)
            return (w @ m) / k  # (n_boot, N)
        dr = seed_resampled_means(m_var) - seed_resampled_means(m_ref)
        prof_r = dr @ g  # (n_boot, P)
        boot2 = np.take_along_axis(prof_r, idx, axis=1).sum(axis=1) / denom
        res["ci2_lo"], res["ci2_hi"] = _ci(boot2)
    return res
