"""Tests of scripts/bigdata_analysis.py on synthetic runs (fake eval CSVs, post_val.npz, post_trainfit.json).

A known effect is planted (MDv64-local rel_vy = 0.9 x B on the full set, 0.95 x B on the reduced set) and must be
recovered with the right sign; missing runs are declared; mismatched file lists are refused. No data set is read.

Run: conda run -n mdf python -m pytest -q tests/test_bigdata_analysis.py
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mdf import data as mdata

REPO = Path(__file__).resolve().parent.parent
FACTOR = {"full": {"B": 1.0, "MDv64-local": 0.90, "MDv64": 0.88},
          "reduced": {"B": 1.0, "MDv64-local": 0.95, "MDv64": 0.94}}
CSV_FIELDS = ["index", "file", "rel_p", "rel_vy", "rel_all", "L1", "abs_err_vy", "abs_tgt_vy"]


def files_for(set_name, n_per_profile=3):
    out = []
    for i, p in enumerate(mdata.grouped_val_profiles()[:20]):
        for k in range(n_per_profile):
            out.append(f"reg/{p}_{1000 + k}_{i}.npz" if set_name == "full" else f"{p}_{1000 + k}_{i}.npz")
            if set_name == "full":
                out.append(f"shear/{p}_{k + 1}_{1000 + k}_{i}.npz")
    return sorted(out)


def make_run(root, name, set_name, label, seed, files, rng, post=True):
    d = root / name
    d.mkdir(parents=True)
    mode = mdata.GROUPED_FULL if set_name == "full" else "grouped"
    (d / "config.json").write_text(json.dumps({
        "status": "finished", "seed": seed, "it_per_s_train": 50.0, "git": {"commit": "abcdef0"},
        "split": {"split_mode": mode, "val_files_sha256": mdata.file_list_sha256(files), "n_train": 1000}}))
    n = len(files)
    base = 0.06 + 0.01 * np.sin(np.arange(n))
    rel_vy = base * FACTOR[set_name][label] * (1 + 0.01 * rng.standard_normal(n))
    tgt = 100.0 + np.arange(n)
    with open(d / "eval_val_samples.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for i, f in enumerate(files):
            w.writerow({"index": i, "file": f, "rel_p": 0.2, "rel_vy": rel_vy[i], "rel_all": 0.03 * FACTOR[set_name][label],
                        "L1": 0.01, "abs_err_vy": rel_vy[i] * tgt[i], "abs_tgt_vy": tgt[i]})
    if set_name == "full" and post:
        reg = {"reg_err_wall": np.stack([rel_vy, rel_vy, rel_vy], 1), "reg_n_wall": np.full(n, 100.0)}
        np.savez_compressed(d / "post_val.npz", meta="{}", files=np.array(files), rel_vy=rel_vy, **reg,
                            div_err_wall=rel_vy / 10, div_err_other=rel_vy / 100)
        (d / "post_trainfit.json").write_text(json.dumps({"train_L1": 0.01 * FACTOR[set_name][label] + 1e-4 * seed,
                                                          "val_L1": 0.012 * FACTOR[set_name][label] + 1e-4 * seed}))


def build(tmp_path, skip=()):
    rng = np.random.default_rng(0)
    full_root, red_root = tmp_path / "full", tmp_path / "reduced"
    for set_name, root in (("full", full_root), ("reduced", red_root)):
        files = files_for(set_name)
        for label in ("B", "MDv64-local", "MDv64"):
            for seed in range(3):
                if (set_name, label, seed) not in skip:
                    make_run(root, f"{label}_s{seed}", set_name, label, seed, files, rng)
    return full_root, red_root


def run(tmp_path, full_root, red_root):
    out = tmp_path / "out"
    cmd = [sys.executable, "scripts/bigdata_analysis.py", "--full-root", str(full_root), "--seeds", "0-2",
           "--n-boot", "500", "--out-dir", str(out), "--fig-dir", str(out)]
    for label in ("B", "MDv64-local", "MDv64"):
        cmd += ["--reduced", f"{label}={red_root}/{label}_s*"]
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True), out


def comparisons(out):
    with open(out / "comparisons.csv") as fh:
        return {(r["set"], r["comparison"], r["metric"]): r for r in csv.DictReader(fh)}


def test_planted_effect_recovered(tmp_path):
    p, out = run(tmp_path, *build(tmp_path))
    assert p.returncode == 0, p.stderr
    c = comparisons(out)
    for s in ("full", "reduced"):
        r = c[(s, "MDv64-local - B", "rel_vy")]
        assert float(r["hi"]) < 0 and r["seeds"] == "0,1,2"
    # full-set effect (10 %) larger than the reduced one (5 %)
    assert float(c[("full", "MDv64-local - B", "rel_vy")]["diff"]) < float(c[("reduced", "MDv64-local - B",
                                                                                 "rel_vy")]["diff"])
    assert float(c[("full", "MDv64-local - B", "wall_vy_mae_x1e-3")]["hi"]) < 0
    assert ("reduced", "MDv64-local - B", "wall_vy_mae_x1e-3") not in c  # full set only
    for metric in ("agg_rel_vy", "median_rel_vy", "rel_all", "rel_p", "div_err_wall_x1e-3", "div_err_other_x1e-3"):
        assert ("full", "MDv64 - B", metric) in c
    assert (out / "bigdata_rel_vy.pdf").exists() and (out / "bigdata_rel_vy.png").exists()
    assert (out / "trainfit_5a.csv").exists()
    text = (out / "summary.md").read_text()
    assert "none missing" in text and "No confirmatory decision" in text


def test_missing_runs_declared(tmp_path):
    p, out = run(tmp_path, *build(tmp_path, skip={("full", "MDv64", 0), ("full", "MDv64", 1), ("full", "MDv64", 2),
                                                   ("full", "B", 2)}))
    assert p.returncode == 0, p.stderr
    c = comparisons(out)
    assert c[("full", "MDv64 - B", "all")]["estimate [95% CI]"] == "not computed: runs missing"
    assert c[("full", "MDv64-local - B", "rel_vy")]["seeds"] == "0,1"  # common seeds only
    text = (out / "summary.md").read_text()
    assert "MDv64 [0, 1, 2]" in text and "B [2]" in text


def test_mismatched_files_refused(tmp_path):
    full_root, red_root = build(tmp_path)
    csv_path = full_root / "B_s1" / "eval_val_samples.csv"
    lines = csv_path.read_text().splitlines()
    csv_path.write_text("\n".join([lines[0], lines[2], lines[1], *lines[3:]]) + "\n")
    p, _ = run(tmp_path, full_root, red_root)
    assert p.returncode != 0 and "differ in validation files" in (p.stdout + p.stderr)


def test_wrong_split_mode_refused(tmp_path):
    full_root, red_root = build(tmp_path)
    cfg = json.loads((full_root / "B_s0" / "config.json").read_text())
    cfg["split"]["split_mode"] = "grouped"
    (full_root / "B_s0" / "config.json").write_text(json.dumps(cfg))
    p, _ = run(tmp_path, full_root, red_root)
    assert p.returncode != 0 and "expected grouped-full" in (p.stdout + p.stderr)
