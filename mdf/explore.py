"""Per-sample / per-region analysis helpers: the statistics behind every comparison in the poster and report.

Three ideas:
  1. Per-sample errors. A trained run is evaluated sample by sample (run_inference): for each sample and channel,
     the relative L1 error sum|o - t| / sum|t| (as train/runTest.py), plus the same sums restricted to flow regions.
     Results are cached on disk (cached_inference), keyed by the checkpoint's sha256, so analyses can be rerun
     without a GPU.
  2. Regions. Where in the flow does a model improve? Every fluid cell is assigned to one region: the wall band
     (1-4 px from the airfoil, a proxy of the boundary layer; the poster's "within 4 px of the airfoil"), the wake
     behind the trailing edge, or the rest.
  3. Paired, two-level bootstrap (paired_bootstrap). Two configurations are compared on the SAME samples with the
     SAME seeds (paired). The uncertainty has two sources: which airfoils happen to be in the evaluation set, and
     which training seeds were run. Each bootstrap repetition resamples both: the profiles (all samples of an
     airfoil together, since they are not independent) and, per configuration, the seeds. The 2.5 / 97.5
     percentiles of the repeated differences give the 95 % intervals in the poster; a pre-registered improvement
     is CONFIRMED only if the whole interval lies below 0.

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

Divergence (docs/lab-notebook.md, 2026-10-14): velocity channels brought back to v / |v_inf| by multiplying by their
fixed maxima (vx 2.04, vy 2.37, mdata.MAX_TARGETS); central differences on the uniform grid (unit: per pixel), only
in fluid cells whose 4 neighbours are fluid. A mapping (vx_axis, sign_x, sign_y) says along which array axis vx is
differentiated (vy along the other one) and with which signs: div = sign_x d(vx)/d(axis) + sign_y d(vy)/d(other).
Flipping both signs only flips the sign of div, so the 8 mappings give 4 distinct mean |div| values.
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
CACHE_VERSION = 3
DIV_REGIONS = ("wall", "other")  # wall band; other = wake + rest (fluid away from the wall)
# axis mapping of the divergence, chosen from validation targets (axis_mapping_scan, notebook 2026-10-14)
DIV_MAPPING = (0, 1, 1)  # vx along axis 0 (rows), same sign: mean |div target| 4.1e-4 vs >= 6.9e-3 otherwise


# --- test-set guard ---
# The test set is opened only for pre-registered final comparisons; everything exploratory uses validation. These
# checks make an accidental read of data/test fail loudly instead of silently leaking test information.

def assert_files_allowed(directory, files, split):
    """Explicit guard: for splits 'val' and 'train' every file must be read from data/train, never from data/test."""
    directory = Path(directory).resolve()
    test_dir = mdata.TEST_DIR.resolve()
    train_dir = mdata.TRAIN_DIR.resolve()
    if split in ("val", "train", "val_seen"):
        assert directory == train_dir, f"val files must come from {train_dir}, got {directory}"
        assert directory != test_dir, "refusing to read the test directory for split 'val'"
        for f in files:
            path = (directory / f).resolve()
            assert test_dir not in path.parents, f"{path} is inside the test directory"
            assert path.parent == train_dir, f"{path} is not in {train_dir}"
    elif split != "test":
        raise ValueError(f"unknown split {split!r}")


def load_split(split, split_mode="grouped", allow_test=False):
    """FlowDataset for 'val' (profile-disjoint by default, checked against the committed split file), 'train' (the
    training part of the same split), 'val_seen' (seen-shape validation, split mode grouped-holdout only) or
    'test'."""
    if split == "test":
        if not allow_test:
            raise SystemExit("split 'test' is for final results only; pass --allow-test explicitly")
        files = mdata.list_files(mdata.TEST_DIR)
        assert_files_allowed(mdata.TEST_DIR, files, "test")
        return mdata.FlowDataset(mdata.TEST_DIR, files)
    if split == "val_seen":  # notebook 2026-10-15, seen-shape vs unseen-shape validation
        if split_mode != mdata.HOLDOUT:
            raise ValueError(f"split 'val_seen' exists only for split mode {mdata.HOLDOUT}")
        train_files, val_files, seen = mdata.split_files_grouped_holdout_parts(mdata.list_files(mdata.TRAIN_DIR))
        if mdata.read_split_file(mdata.seen_split_file_path()) != seen:
            raise RuntimeError("computed seen-shape validation differs from the committed split file")
        assert not set(seen) & (set(train_files) | set(val_files)), "seen-shape validation overlaps"
        assert_files_allowed(mdata.TRAIN_DIR, seen, "val_seen")
        return mdata.FlowDataset(mdata.TRAIN_DIR, seen)
    train_files, val_files = mdata.SPLIT_MODES[split_mode](mdata.list_files(mdata.TRAIN_DIR))
    if mdata.read_split_file(mdata.split_file_path(split_mode)) != val_files:
        raise RuntimeError("computed validation split differs from the committed split file")
    if split == "train":  # training part of the split (training-error check, notebook 2026-10-14)
        assert not set(train_files) & set(val_files), "train and validation files overlap"
        assert_files_allowed(mdata.TRAIN_DIR, train_files, "train")
        return mdata.FlowDataset(mdata.TRAIN_DIR, train_files)
    if split != "val":
        raise ValueError(f"unknown split {split!r}")
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


# --- divergence ---

ALL_MAPPINGS = tuple((ax, sx, sy) for ax in (0, 1) for sx in (1, -1) for sy in (1, -1))


def valid_stencil(fluid):
    """Fluid cells (bool (..., H, W)) whose 4 neighbours are fluid; grid-edge cells are excluded."""
    v = np.zeros_like(fluid, dtype=bool)
    v[..., 1:-1, 1:-1] = (fluid[..., 1:-1, 1:-1] & fluid[..., :-2, 1:-1] & fluid[..., 2:, 1:-1]
                          & fluid[..., 1:-1, :-2] & fluid[..., 1:-1, 2:])
    return v


def divergence(vx, vy, mapping, scale=(mdata.MAX_TARGETS[1], mdata.MAX_TARGETS[2])):
    """Divergence (per pixel) of (vx, vy) arrays (..., H, W) in normalised units, multiplied by `scale` first.
    Central differences; values on the outer ring are 0 and must be masked with valid_stencil."""
    ax, sx, sy = mapping
    vx = np.asarray(vx, dtype=np.float64) * scale[0]
    vy = np.asarray(vy, dtype=np.float64) * scale[1]

    def d(f, axis):  # central difference along array axis 0 (rows) or 1 (cols) of the last two dims
        out = np.zeros_like(f)
        if axis == 0:
            out[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / 2.0
        else:
            out[..., :, 1:-1] = (f[..., :, 2:] - f[..., :, :-2]) / 2.0
        return out

    return sx * d(vx, ax) + sy * d(vy, 1 - ax)


def axis_mapping_scan(ds):
    """Mean over samples of mean |div target| over valid cells, for all 8 mappings (model independent)."""
    tgt = ds.targets.numpy()
    valid = valid_stencil(ds.inputs[:, 2].numpy() < 0.5)
    out = {}
    for mp in ALL_MAPPINGS:
        a = np.abs(divergence(tgt[:, 1], tgt[:, 2], mp))
        out[mp] = float(np.mean((a * valid).sum(axis=(1, 2)) / valid.sum(axis=(1, 2))))
    return out


def div_regions(regions_i, valid_i):
    return {"wall": regions_i["wall"] & valid_i, "other": (regions_i["wake"] | regions_i["rest"]) & valid_i}


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
      div_pred_<r>, div_tgt_<r>, div_err_<r>, div_n_<r>   (N,) mean |div pred|, mean |div target|,
                    mean |div pred - div target| over the valid cells of r in DIV_REGIONS (DIV_MAPPING), and count
      mean_abs_err_map  (3, 128, 128) mean over samples of |o - t|
      keep_pred (K, 3, 128, 128) float32, keep_idx (K,)
    """
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    mcfg = models.model_cfg_from_run_config(cfg)
    if mcfg["variant"] in ("MD", "MDv") and device == "cpu":
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
    if DIV_MAPPING is None:
        raise RuntimeError("DIV_MAPPING not set")
    for r in DIV_REGIONS:
        for k in ("pred", "tgt", "err", "n"):
            out[f"div_{k}_{r}"] = np.zeros(n)
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
                dp = divergence(o[1], o[2], DIV_MAPPING)
                dt = divergence(t[1], t[2], DIV_MAPPING)
                valid = valid_stencil(ds.inputs[i, 2].numpy() < 0.5)
                for r, m in div_regions(regions[i], valid).items():
                    cnt = max(int(m.sum()), 1)
                    out[f"div_pred_{r}"][i] = np.abs(dp[m]).sum() / cnt
                    out[f"div_tgt_{r}"][i] = np.abs(dt[m]).sum() / cnt
                    out[f"div_err_{r}"][i] = np.abs(dp[m] - dt[m]).sum() / cnt
                    out[f"div_n_{r}"][i] = m.sum()
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
        "div_mapping": list(DIV_MAPPING) if DIV_MAPPING else None,
    }
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_hash = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    path = cache_dir / f"{run_dir.parent.name}__{run_dir.name}__{split}__{key_hash}.npz"
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

    Implementation: profile membership is a one-hot matrix g (N samples x P profiles), so "sum of the differences
    of the resampled profiles" is a matrix product, and all n_boot repetitions are computed at once without a loop.
    ci1_* = one-level interval, ci2_* = two-level interval (the one reported).
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


def _weighted_median(values, weights):
    """Median of `values` repeated `weights` times (integer weights), as np.median of the repeated array."""
    return float(np.median(np.repeat(values, weights.astype(np.int64))))


def paired_bootstrap_agg_median(err_var, err_ref, tgt, rel_var, rel_ref, profiles, n_boot=10000, seed=0):
    """Paired differences var - ref of the aggregate ratio and of the median per-sample ratio of one channel.

    err_*: (seeds, N) per-sample sum|o - t|; tgt: (N,) per-sample sum|t| (same for every run); rel_*: (seeds, N)
    per-sample ratios. Point estimates on seed means: aggregate = sum_s err / sum_s tgt; median = median over samples of
    the seed-mean per-sample ratio. Intervals: one level (profiles resampled with replacement; a sample counts as many
    times as its profile is drawn) and two levels (also the seeds of each configuration resampled), as
    paired_bootstrap."""
    err_var, err_ref = np.atleast_2d(err_var), np.atleast_2d(err_ref)
    rel_var, rel_ref = np.atleast_2d(rel_var), np.atleast_2d(rel_ref)
    g, _ = profile_onehot(profiles)
    n_prof = g.shape[1]
    sample_prof = g.argmax(axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_prof, size=(n_boot, n_prof))
    prof_counts = np.zeros((n_boot, n_prof))
    np.add.at(prof_counts, (np.arange(n_boot)[:, None], idx), 1.0)
    w = prof_counts[:, sample_prof]  # (n_boot, N) multiplicity of each sample

    def seed_weights(k):  # the same resampled seeds are used for err and rel of one configuration
        pick = rng.integers(0, k, size=(n_boot, k))
        sw = np.zeros((n_boot, k))
        np.add.at(sw, (np.arange(n_boot)[:, None], pick), 1.0)
        return sw / k

    def stats(ev, er, rv, rr, weights):
        agg = (weights * ev).sum(-1) / (weights * tgt).sum(-1) - (weights * er).sum(-1) / (weights * tgt).sum(-1)
        if rv.ndim == 1:
            med = np.array([_weighted_median(rv, wi) - _weighted_median(rr, wi) for wi in weights])
        else:
            med = np.array([_weighted_median(a, wi) - _weighted_median(b, wi) for a, b, wi in zip(rv, rr, weights)])
        return agg, med

    ev, er, rv, rr = err_var.mean(0), err_ref.mean(0), rel_var.mean(0), rel_ref.mean(0)
    res = {"agg_diff": float(ev.sum() / tgt.sum() - er.sum() / tgt.sum()),
           "median_diff": float(np.median(rv) - np.median(rr)),
           "seeds_var": int(err_var.shape[0]), "seeds_ref": int(err_ref.shape[0])}
    a1, m1 = stats(ev, er, rv, rr, w)
    sv, sr = seed_weights(err_var.shape[0]), seed_weights(err_ref.shape[0])
    a2, m2 = stats(sv @ err_var, sr @ err_ref, sv @ rel_var, sr @ rel_ref, w)
    for name, arr in (("agg_ci1", a1), ("median_ci1", m1), ("agg_ci2", a2), ("median_ci2", m2)):
        res[name + "_lo"], res[name + "_hi"] = _ci(arr)
    return res
