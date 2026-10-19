"""Checks of the unpacked full data set (cloud VM; docs/cloud-guide.md step 5, docs/preregistration-bigdata.md).

File names only, unless --load-ram is given. The test set is not unpacked on the VM and never read; the test profiles
come from splits/test_profiles.txt (and, with --archive-test-names, from the test names listed in the archive).

Reports, for data/train_full (symlink to the unpacked data/train/):
  - entries per subdirectory (reg, shear, sup, anything else) and names that do not parse (profile_full rules);
  - samples per base profile (min / median / max), number of profiles, profiles present only in shear;
  - the 85 grouped validation profiles and the 30 test profiles: files per subdirectory;
  - near names: other profiles whose name starts with a validation / test profile followed by "_" (descriptive);
  - overlap with the 6.4k set: how many of the 402 grouped validation files of the 6.4k set exist in reg/;
  - the grouped-full split for the subdirectories in use ($MDF_FULL_SUBDIRS, default reg,shear): sizes and sha256.
--load-ram: also loads the training and validation sets with the training loader (mdf.data.load_train_val_full)
and prints the peak RSS (no GPU), to check that the VM has enough RAM before any run.
Writes the report as JSON with --json FILE. Exit code 1 if a name of a used subdirectory does not parse, if a test
profile would reach train or validation, or if a validation profile is missing.

Usage:
    python scripts/cloud/inspect_full.py [--archive-test-names FILE] [--load-ram] [--json FILE]
"""

import argparse
import json
import resource
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root, for `mdf`

from mdf import data as mdata


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full-dir", type=Path, default=mdata.FULL_DIR)
    ap.add_argument("--archive-test-names", type=Path, default=None,
                    help="file with the data/test/*.npz names listed in the archive (get_full_data.sh)")
    ap.add_argument("--load-ram", action="store_true", help="load train + val with the training loader, peak RSS")
    ap.add_argument("--json", type=Path, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    full = args.full_dir
    if not full.is_dir():
        sys.exit(f"{full} not found (symlink data/train_full -> unpacked data/train/, see get_full_data.sh)")
    report, problems = {"full_dir": str(full.resolve())}, []

    entries = sorted(p for p in full.iterdir())
    report["top_level"] = {p.name: ("dir" if p.is_dir() else "file") for p in entries}
    per_sub, bad, prof_of = {}, defaultdict(list), {}
    for d in entries:
        if not d.is_dir():
            continue
        names = sorted(p.name for p in d.iterdir())
        per_sub[d.name] = {"entries": len(names), "npz": sum(n.endswith(".npz") for n in names)}
        for n in names:
            rel = f"{d.name}/{n}"
            try:
                prof_of[rel] = mdata.profile_full(rel)
            except ValueError:
                bad[d.name].append(n)
    report["per_subdir"] = per_sub
    report["unparsed_per_subdir"] = {s: {"count": len(v), "examples": v[:10]} for s, v in bad.items()}
    print(f"{full}: top level {report['top_level']}")
    for s, c in per_sub.items():
        print(f"  {s:8s} entries {c['entries']:6d}  .npz {c['npz']:6d}  not parsed {len(bad.get(s, [])):6d}"
              + (f"  e.g. {bad[s][:3]}" if bad.get(s) else ""))

    subs = mdata.full_subdirs()
    report["subdirs_used"] = list(subs)
    for s in subs:
        if bad.get(s):
            problems.append(f"{len(bad[s])} names in used subdir {s}/ do not parse")
    print(f"Subdirs used for grouped-full: {subs} (MDF_FULL_SUBDIRS to change; declare it in the notebook)")

    by_sub_prof = defaultdict(Counter)
    for rel, p in prof_of.items():
        by_sub_prof[rel.split("/")[0]][p] += 1
    stats = {}
    for s, cnt in sorted(by_sub_prof.items()):
        v = list(cnt.values())
        stats[s] = {"profiles": len(cnt), "files_per_profile_min": min(v), "median": statistics.median(v),
                    "max": max(v)}
        print(f"  {s:8s} profiles {len(cnt):5d}  files/profile min {min(v)} median {statistics.median(v)} "
              f"max {max(v)}")
    report["profiles_per_subdir"] = stats
    if "reg" in by_sub_prof and "shear" in by_sub_prof:
        only_shear = sorted(set(by_sub_prof["shear"]) - set(by_sub_prof["reg"]))
        report["profiles_only_in_shear"] = {"count": len(only_shear), "examples": only_shear[:10]}
        print(f"  profiles only in shear (no regular sample): {len(only_shear)} {only_shear[:5]}")

    val_p, test_p = mdata.grouped_val_profiles(), mdata.profiles_of_test()
    if args.archive_test_names is not None:
        names = mdata.read_split_file(args.archive_test_names)
        arch_p = sorted({mdata.profile(Path(n).name) for n in names})
        report["archive_test"] = {"files": len(names), "profiles": len(arch_p), "same_as_committed": arch_p == test_p}
        print(f"Archive data/test: {len(names)} files, {len(arch_p)} profiles; equal to splits/test_profiles.txt: "
              f"{arch_p == test_p}")
        if arch_p != test_p:
            problems.append("test profiles in the archive differ from splits/test_profiles.txt")
    all_p = set(prof_of.values())
    for label, group in (("validation", val_p), ("test", test_p)):
        counts = {s: sum(by_sub_prof[s][p] for p in group) for s in by_sub_prof}
        present = sorted(set(group) & all_p)
        near = sorted({q for q in all_p - set(group) for p in group if q.startswith(p + "_")})
        report[f"{label}_profiles"] = {"n": len(group), "present": len(present), "files_per_subdir": counts,
                                       "near_names": near[:50], "n_near_names": len(near)}
        print(f"{label} profiles: {len(present)} of {len(group)} present; files per subdir {counts}; "
              f"near names (other profiles starting with '<profile>_'): {len(near)} {near[:5]}")
    six_k_val = mdata.read_split_file(mdata.split_file_path("grouped"))
    in_reg = sum((full / "reg" / f).exists() for f in six_k_val)
    report["grouped_val_6k_files_in_reg"] = in_reg
    print(f"6.4k grouped validation files found in reg/: {in_reg} of {len(six_k_val)}")

    files = mdata.list_files_full(full, subs)
    try:
        train, val, excl = mdata.split_files_grouped_full(files, val_p, test_p)
    except ValueError as e:
        problems.append(str(e))
        train = val = excl = []
    tr_p = {mdata.profile_full(f) for f in train}
    va_p = {mdata.profile_full(f) for f in val}
    if (tr_p | va_p) & set(test_p):
        problems.append("a test profile reaches train or validation")
    if va_p != set(val_p):
        problems.append(f"validation profiles missing in the full set: {sorted(set(val_p) - va_p)}")
    report["split"] = {"n_files": len(files), "n_train": len(train), "n_val": len(val), "n_excluded_test": len(excl),
                       "n_train_profiles": len(tr_p), "n_val_profiles": len(va_p),
                       "train_files_sha256": mdata.file_list_sha256(train),
                       "val_files_sha256": mdata.file_list_sha256(val)}
    print(f"grouped-full ({','.join(subs)}): files {len(files)}, train {len(train)} ({len(tr_p)} profiles), "
          f"val {len(val)} ({len(va_p)} profiles), excluded (test profiles) {len(excl)}")
    print(f"  val sha256 {report['split']['val_files_sha256']}")
    bytes_per_sample = 2 * 3 * 128 * 128 * 4
    est = (len(train) + len(val)) * bytes_per_sample / 2**30
    report["ram_estimate_gib_data"] = est
    print(f"RAM for the loaded arrays (float32, {bytes_per_sample} B/sample): {est:.1f} GiB "
          f"(+ ~1.4 GiB process overhead measured on the 6.4k runs)")

    if args.load_ram and not problems:
        t0 = time.perf_counter()
        tr_ds, va_ds, _ = mdata.load_train_val_full(full)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        report["load_ram"] = {"n_train": len(tr_ds), "n_val": len(va_ds), "seconds": time.perf_counter() - t0,
                              "peak_rss_gib": peak}
        print(f"Loaded train {len(tr_ds)} + val {len(va_ds)} in {time.perf_counter() - t0:.0f} s; "
              f"peak RSS {peak:.1f} GiB (no CUDA context; training adds ~1-2 GiB)")

    report["problems"] = problems
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {args.json}")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
