"""Poster figures (DML poster, A2 portrait; docs/lab-notebook.md, 2026-10-20). Vector PDFs in docs/figures/poster/.

Validation only. No test inference and no evaluate.py: test numbers are read from docs/results/test-* only. Per-seed
validation values come from the explore cache (~/runs/explore-cache, cache hits only: a miss stops the script).

  fig_level.pdf       22.4 x 6.5 cm (right column). rel_vy vs encoder block level 8 / 16 / 32 / 64 (MDv), seeds + mean;
                      B mean +- sd band; M64-dec (same data as val-ablation64-8seed/ablation64_rel_vy panel (b))
  fig_ablation.pdf    22.4 x 8.5 cm (right column). Forest plot of rel_vy - B with the two-level 95 % CI already
                      computed (ci2_lo / ci2_hi), validation open markers, test filled markers (M64, M64-loc); band =
                      M64 - B validation interval
  poster_numbers.csv  every value shown on fig_level and fig_ablation (written when both are drawn)
  fig_wallmap         NOT written unless the validation airfoil masks are identical across samples (they are not:
                      the cached mean |error| maps average different airfoils; the check is printed)
  fig_tail_val.pdf    25 x 8.2 cm, only with --only tail (kept as first produced on 2026-10-20). p of one separated-flow
                      validation sample: target, B, MD8 (cached predictions; no test sample and no M64 prediction is
                      cached, see the notebook entry)

Sans-serif Liberation Sans (Helvetica metrics). The TikZ block diagram is docs/figures/poster/fig_block.tex (16.2 x
6.3 cm, left column; hand-written, not generated here).

Usage: python scripts/poster_figures.py [--out-dir docs/figures/poster] [--only level,ablation,wallmap,tail]
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from mdf import explore

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "docs" / "results"
CACHE = Path.home() / "runs" / "explore-cache"
RUNS_CSV = RES / "val-ablation64-8seed" / "runs.csv"
CM = 1 / 2.54

# sizes measured on the poster layout (body 16 pt, captions 9.4 pt; columns 16.2 / 22.4 cm)
SIZE_LEVEL, SIZE_ABLATION, SIZE_TAIL = (22.4 * CM, 6.5 * CM), (22.4 * CM, 8.5 * CM), (25 * CM, 8.2 * CM)
RC_LEVEL = {"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14}
RC_ABLATION = {"font.size": 13, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
               "legend.fontsize": 13}
RC_TAIL = {"font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18}

# one palette for every figure (dataviz reference palette, validated: blue / orange / aqua, light mode)
C_M64, C_LOC, C_DEC = "#2a78d6", "#eb6834", "#1baf7a"
C_B, C_OTHER, C_INK, C_GRID = "#8a8a85", "#4a4a47", "#1f1f1e", "#e6e5e1"
DIVERGING = LinearSegmentedColormap.from_list("poster_div", ["#2a78d6", "#f0efec", "#e34948"])
C_MASK = "#b4b3ad"

POSTER_NAME = {"B": "B", "MDv64": "M64", "MDv64-local": "M64-loc", "SA64": "SA64", "Conv64": "Conv64",
               "MDv64-dec": "M64-dec", "MDv8": "M8", "MDv16": "M16", "MDv32": "M32", "B-large": "B-large",
               "L-nogate": "M64-loc −gate", "L-nonorm": "M64-loc −out-norm", "L-noprenorm": "M64-loc −pre-norm",
               "L-noexp": "M64-loc −expansion", "L-dense": "M64-loc dense"}

NUMBER_FIELDS = ["figure", "kind", "poster_name", "label", "split", "seed", "unit", "estimate", "ci_lo", "ci_hi",
                 "interval", "n_seeds_var", "n_seeds_ref", "source"]
NUMBERS = []  # rows of poster_numbers.csv


def style():
    # Liberation Sans: TrueType with Helvetica metrics (matplotlib embeds the OTF/CFF Nimbus Sans with a font-type
    # mismatch that pdffonts flags), embedded as Type 42. Font sizes per figure (RC_*).
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
        "axes.edgecolor": C_INK, "axes.labelcolor": C_INK, "xtick.color": C_INK, "ytick.color": C_INK,
        "text.color": C_INK, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "mathtext.default": "regular",
        "savefig.transparent": False, "figure.facecolor": "white",
    })


def commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def save(fig, out_dir, name):
    path = out_dir / name
    fig.savefig(path, metadata={"Creator": f"scripts/poster_figures.py @ {commit()}"})
    plt.close(fig)
    print(f"wrote {path}")


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def rel(path):
    return str(Path(path).relative_to(ROOT))


def number(**kw):
    NUMBERS.append({k: kw.get(k, "") for k in NUMBER_FIELDS})


def write_numbers(out_dir):
    fmt = lambda v: f"{v:.6f}" if isinstance(v, float) else v
    path = out_dir / "poster_numbers.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, NUMBER_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows({k: fmt(v) for k, v in r.items()} for r in NUMBERS)
    print(f"wrote {path} ({len(NUMBERS)} rows)")


def paired(path, comparison, metric="rel_vy"):
    """(mean, ci2_lo, ci2_hi, seeds_var, seeds_ref) in percentage points; ci2 = two-level bootstrap interval."""
    for r in rows(path):
        if r["comparison"] == comparison and r["metric"] == metric:
            return (100 * float(r["mean_diff"]), 100 * float(r["ci2_lo"]), 100 * float(r["ci2_hi"]),
                    int(r["seeds_var"]), int(r["seeds_ref"]))
    raise KeyError(f"{comparison} / {metric} not in {path}")


def cached(run_dir, ds, keep_idx=()):
    """Cached validation inference of one run; a cache miss stops the script (no new inference here)."""
    try:
        res, hit = explore.cached_inference(Path(run_dir).expanduser(), ds, None, list(keep_idx), CACHE, "val",
                                            device="cpu")
    except TypeError:  # regions=None reached run_inference: not cached
        hit = False
    if not hit:
        raise SystemExit(f"{run_dir}: no validation cache entry (keep {list(keep_idx)}); refusing to infer")
    return res


def per_seed_rel_vy(ds, labels):
    """{label: (seeds, rel_vy %)} for the runs listed in RUNS_CSV (same runs as the ablation figure)."""
    out = {}
    for r in rows(RUNS_CSV):
        if r["label"] in labels:
            v = 100 * cached(r["run_dir"], ds)["rel_vy"].mean()
            out.setdefault(r["label"], []).append((int(r["seed"]), v))
    return {k: (np.array([s for s, _ in v]), np.array([x for _, x in v])) for k, v in out.items()}


def fig_level(ds, out_dir):
    levels = [("MDv8", 8), ("MDv16", 16), ("MDv32", 32), ("MDv64", 64)]
    data = per_seed_rel_vy(ds, [l for l, _ in levels] + ["B", "MDv64-dec"])
    src = f"explore cache (val), runs of {rel(RUNS_CSV)}"
    print("fig_level: rel_vy %, mean (sd, ddof=1), n")
    for k, (s, v) in data.items():
        print(f"  {k:10s} {v.mean():.4f} ({v.std(ddof=1):.4f}) n={v.size}  seeds {s.tolist()}")
        sd = v.std(ddof=1)
        number(figure="fig_level", kind="mean", poster_name=POSTER_NAME[k], label=k, split="val", unit="%",
               estimate=v.mean(), ci_lo=v.mean() - sd if k == "B" else "", ci_hi=v.mean() + sd if k == "B" else "",
               interval="mean +- sd (ddof=1), band" if k == "B" else "", n_seeds_var=v.size, source=src)
        for seed, x in zip(s, v):
            number(figure="fig_level", kind="seed", poster_name=POSTER_NAME[k], label=k, split="val", seed=int(seed),
                   unit="%", estimate=x, n_seeds_var=1, source=src)
    with plt.rc_context(RC_LEVEL):
        fig, ax = plt.subplots(figsize=SIZE_LEVEL, constrained_layout=True)
        b = data["B"][1]
        ax.axhspan(b.mean() - b.std(ddof=1), b.mean() + b.std(ddof=1), color=C_B, alpha=0.18, lw=0, zorder=0)
        ax.axhline(b.mean(), color=C_B, lw=1.2, ls="--", zorder=1)
        ax.text(5.3, b.mean() + 0.03, f"B  (mean ± sd, n = {b.size})", color=C_OTHER, va="bottom")
        xs = np.log2([lv for _, lv in levels])
        means = []
        for (l, lv), x in zip(levels, xs):
            v = data[l][1]
            jit = (np.arange(v.size) - (v.size - 1) / 2) * 0.035
            ax.scatter(x + jit, v, s=26, color=C_M64, alpha=0.55, edgecolors="none", zorder=2)
            means.append(v.mean())
        ax.plot(xs, means, color=C_M64, lw=1.6, marker="o", ms=7, mec="white", mew=1.2, zorder=3)
        d = data["MDv64-dec"][1]
        x_dec = np.log2(64) + 0.3
        jit = (np.arange(d.size) - (d.size - 1) / 2) * 0.035
        ax.scatter(x_dec + jit, d, s=26, marker="D", color=C_DEC, alpha=0.6, edgecolors="none", zorder=2)
        ax.plot([x_dec], [d.mean()], marker="D", ms=8, color=C_DEC, mec="white", mew=1.2, zorder=3, ls="none")
        ax.annotate(f"M64-dec\n(decoder, n = {d.size})", (x_dec, d.mean()), xytext=(10, 0),
                    textcoords="offset points", ha="left", va="center", color=C_INK)
        ax.set_xticks(xs, [f"{s}×{s}\n{POSTER_NAME[l]}, n = {data[l][1].size}" for l, s in levels])
        ax.set_xlim(2.5, 7.25)
        ax.set_xlabel("feature-map size of the encoder block")
        ax.set_ylabel("rel. error $v_y$ (%)")
        ax.grid(axis="y", color=C_GRID, lw=0.8)
        ax.set_axisbelow(True)
        save(fig, out_dir, "fig_level.pdf")


def fig_ablation(out_dir):
    ab, fin = RES / "val-ablation64-8seed" / "t2_paired.csv", RES / "val-final-mdv64local" / "t2_paired.csv"
    spec = [  # (variant label, validation source, comparison)
        ("MDv64", ab, "MDv64 - B"),
        ("MDv64-local", fin, "MDv64-local - B"),
        ("SA64", ab, "SA64 - B"),
        *[(l, ab, f"{l} - B") for l in ("L-nogate", "L-nonorm", "L-noprenorm", "L-noexp", "L-dense")],
        ("Conv64", ab, "Conv64 - B"),
        ("MDv64-dec", ab, "MDv64-dec - B"),
        ("MDv16", ab, "MDv16 - B"),
        ("B-large", RES / "val-explore-8seed" / "a3_paired.csv", "B-large - B"),
    ]
    test_src = {"MDv64": RES / "test-confirm-mdv64" / "t2_paired.csv", "MDv64-local": RES / "test-final" / "t2_paired.csv"}
    test = {l: paired(p, f"{l} - B") for l, p in test_src.items()}
    color = {"MDv64": C_M64, "MDv64-local": C_LOC, "MDv64-dec": C_DEC}
    ci = "two-level bootstrap 95 % (ci2), rel_vy - B"
    print("fig_ablation: rel_vy - B, points, two-level 95 % CI (seeds var / ref)")
    with plt.rc_context(RC_ABLATION):
        fig, ax = plt.subplots(figsize=SIZE_ABLATION, constrained_layout=True)
        band = paired(ab, "MDv64 - B")
        number(figure="fig_ablation", kind="band", poster_name="M64", label="MDv64", split="val", unit="points",
               estimate=band[0], ci_lo=band[1], ci_hi=band[2], interval=ci, n_seeds_var=band[3], n_seeds_ref=band[4],
               source=rel(ab))
        ax.axvspan(band[1], band[2], color=C_M64, alpha=0.12, lw=0, zorder=0)
        ax.axvline(0, color=C_INK, lw=0.9, zorder=1)
        n = len(spec)
        for i, (label, src, comp) in enumerate(spec):
            y = n - 1 - i
            m, lo, hi, sv, sr = paired(src, comp)
            print(f"  val  {label:12s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({sv} vs {sr})  {rel(src)}")
            number(figure="fig_ablation", kind="diff", poster_name=POSTER_NAME[label], label=label, split="val",
                   unit="points", estimate=m, ci_lo=lo, ci_hi=hi, interval=ci, n_seeds_var=sv, n_seeds_ref=sr,
                   source=rel(src))
            c = color.get(label, C_OTHER)
            dy = 0.2 if label in test else 0.0
            ax.plot([lo, hi], [y + dy] * 2, color=c, lw=1.7, solid_capstyle="round", zorder=2)
            ax.plot([m], [y + dy], marker="o", ms=6.5, mfc="white", mec=c, mew=1.7, zorder=3)
            if label in test:
                tm, tlo, thi, tsv, tsr = test[label]
                print(f"  test {label:12s} {tm:+.3f} [{tlo:+.3f}, {thi:+.3f}]  ({tsv} vs {tsr})  {rel(test_src[label])}")
                number(figure="fig_ablation", kind="diff", poster_name=POSTER_NAME[label], label=label, split="test",
                       unit="points", estimate=tm, ci_lo=tlo, ci_hi=thi, interval=ci, n_seeds_var=tsv,
                       n_seeds_ref=tsr, source=rel(test_src[label]))
                ax.plot([tlo, thi], [y - dy] * 2, color=c, lw=1.7, solid_capstyle="round", zorder=2)
                ax.plot([tm], [y - dy], marker="o", ms=6.5, mfc=c, mec=c, mew=1.7, zorder=3)
        ax.set_yticks(range(n), [POSTER_NAME[l] for l, _, _ in reversed(spec)])
        for t, (l, _, _) in zip(ax.get_yticklabels(), reversed(spec)):
            t.set_color(color.get(l, C_INK))
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlabel("rel. error $v_y$ − B (percentage points)")
        ax.grid(axis="x", color=C_GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.plot([], [], marker="o", ms=6.5, mfc="white", mec=C_OTHER, mew=1.7, ls="none", label="validation")
        ax.plot([], [], marker="o", ms=6.5, mfc=C_OTHER, mec=C_OTHER, ls="none", label="test")
        ax.legend(loc="lower left", frameon=False, handletextpad=0.2, borderaxespad=0.2)
        save(fig, out_dir, "fig_ablation.pdf")


def fig_wallmap(ds, out_dir):
    """Averaging the cached mean |error| maps is meaningful only if every sample has the same airfoil mask."""
    m = ds.inputs[:, 2].numpy() > 0.5
    freq = m.mean(axis=0)
    first = np.array([np.nonzero(x.any(1))[0][[0, -1]] for x in m])
    cols = np.array([np.nonzero(x.any(0))[0][[0, -1]] for x in m])
    print(f"fig_wallmap: {len(m)} validation samples; cells solid in every sample {int((freq == 1).sum())}, in some "
          f"but not all {int(((freq > 0) & (freq < 1)).sum())}; leading-edge row {first[:, 0].min()}-{first[:, 0].max()}, "
          f"trailing-edge row {first[:, 1].min()}-{first[:, 1].max()}, airfoil columns {cols[:, 0].min()}-"
          f"{cols[:, 0].max()} (first) / {cols[:, 1].min()}-{cols[:, 1].max()} (last); wall band {explore.WALL_PX} px")
    if not (freq[freq > 0] == 1).all():
        print("fig_wallmap: NOT written: the masks differ across samples (different airfoils, not rotated with "
              "alpha), so the cached sample-mean |error| maps mix wall, interior and far-field cells")
        return
    raise NotImplementedError("aligned masks: not expected for this data set")


def fig_tail(ds, out_dir):
    """p of one separated-flow validation sample: target, B and MD8 (median seed on that sample), shared scale."""
    keep = [338, 349, 51, 221, 216, 192]  # samples kept in the cache by analyze_val.py (a7 figures)
    tgt = ds.targets.numpy()
    m_all = ds.inputs[:, 2].numpy() > 0.5
    rev = []
    for i in keep:
        te = np.nonzero(m_all[i].any(1))[0][-1]
        rev.append(int(((tgt[i, 1] < 0) & ~m_all[i])[te + 1:].sum()))
    print("fig_tail: reverse-flow cells downstream of the trailing edge (target vx < 0) per kept sample:")
    for i, r in zip(keep, rev):
        print(f"  {i:4d} {ds.files[i]:28s} {r}")
    i = keep[int(np.argmax(rev))]
    k = keep.index(i)
    R = Path.home() / "runs"
    configs = {"B": [R / "night1" / f"ep133_s{s}" for s in range(3)] + [R / "night3" / f"B_s{s}" for s in range(3, 8)],
               "MD8": [R / "night2" / f"MD8_s{s}" for s in range(3)] + [R / "night3" / f"MD8_s{s}" for s in range(3, 8)]}
    panels = [("target", tgt[i, 0])]
    for label, dirs in configs.items():
        res = [cached(d, ds, keep) for d in dirs]
        e = np.array([r["rel_p"][i] for r in res])
        s = int(np.argsort(e)[(len(e) - 1) // 2])  # lower median seed on this sample
        print(f"  {label}: rel_p on sample {i} per seed {np.round(100 * e, 2).tolist()}; shown s{s}")
        panels.append((f"{label} (s{s})", res[s]["keep_pred"][k, 0].astype(np.float64)))
    mask = m_all[i]
    x0, x1, y0, y1 = 22, 122, 28, 100  # rows = x (flow direction), columns = y (DIV_MAPPING axis 0)
    lim = max(np.abs(p[~mask]).max() for _, p in panels)
    plt.rcParams.update(RC_TAIL)  # sizes of the version kept on the poster (2026-10-20)
    fig, axes = plt.subplots(1, 3, figsize=SIZE_TAIL, constrained_layout=True)
    for ax, (name, p) in zip(axes, panels):
        img = np.ma.masked_where(mask, p)[x0:x1, y0:y1].T  # transpose: flow left -> right, y up
        ax.set_facecolor(C_MASK)
        im = ax.imshow(img, origin="lower", cmap=DIVERGING, vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.03, 0.97, name, transform=ax.transAxes, ha="left", va="top", fontsize=18)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("p (normalised)")
    cb.outline.set_visible(False)
    print(f"  sample {i} {ds.files[i]}; colour range ±{lim:.3f}")
    save(fig, out_dir, "fig_tail_val.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "figures" / "poster")
    ap.add_argument("--only", default="level,ablation,wallmap", help="comma list of level, ablation, wallmap, tail")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    style()
    only = set(args.only.split(","))
    ds = explore.load_split("val", "grouped") if only & {"level", "wallmap", "tail"} else None
    if "level" in only:
        fig_level(ds, args.out_dir)
    if "ablation" in only:
        fig_ablation(args.out_dir)
    if {"level", "ablation"} <= only:
        write_numbers(args.out_dir)
    else:
        print("poster_numbers.csv not written (needs both level and ablation)")
    if "wallmap" in only:
        fig_wallmap(ds, args.out_dir)
    if "tail" in only:
        fig_tail(ds, args.out_dir)


if __name__ == "__main__":
    main()
