"""Sequential run queue with a wall-clock cutoff (night runs; docs/lab-notebook.md, 2026-10-11).

Queue file: one run per line, `#` starts a comment:
    <name> [est=<N>m|<N>s] <scripts/train.py arguments, without --out-dir>
e.g.  ep133_s0  est=13m  --seed 0 --epochs 133

Each run goes to <out-root>/<name>/ (train.py stdout in <name>.log), is evaluated on validation right
after training (scripts/evaluate.py --split val, stdout in <name>.eval_val.log), and its wall times are
written to <name>/queue_timing.json. One run at a time; no resuming in the middle of a run.
Runs trained with split mode grouped-holdout are also evaluated on the seen-shape validation (--split val_seen,
stdout in <name>.eval_val_seen.log); they count as evaluated only when eval_val_seen.json exists too.

State of <out-root>/<name>/ when its line is reached:
  config.json status finished + eval_val.json   -> skipped
  status finished, no eval_val.json             -> evaluation only
  status running / interrupted, no or unreadable config.json
                                                -> interrupted: the directory (and its log) is moved to
                                                   <name>.interrupted-<UTC> and the run is redone from scratch
A training process that exits non-zero gets status "interrupted" (with the exit code); the queue goes on.

Cutoff: the next occurrence of --cutoff HH:MM (local time) after the queue starts, or an absolute local date-time
--cutoff YYYY-MM-DDTHH:MM (added 2026-10-17 for night 9, launched before the cutoff hour on the day before; it must lie
after the start). A run starts only if
now + estimated duration <= cutoff, otherwise the queue stops (later runs are not tried). Estimated
duration = max wall time (train + eval) of finished runs under <out-root> with the same train.py arguments
apart from --seed (from their queue_timing.json); otherwise the est= of the line; with neither, the run
does not start and the queue stops.

Stopping: `touch <out-root>/STOP` stops the queue after the current run (the file is then renamed to
STOP.consumed-<UTC>); SIGINT / SIGTERM terminate the current run (status interrupted) and the queue.

Usage:
    python scripts/run_queue.py QUEUE_FILE --out-root DIR [--cutoff 07:00]
"""

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable

_child = None
_stop_signal = None


def now_local():
    return datetime.now().astimezone()


def stamp(dt=None):
    return (dt or now_local()).isoformat(timespec="seconds")


def utc_tag():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class Log:
    def __init__(self, path):
        self.path = path

    def __call__(self, msg):
        line = f"[{stamp()}] {msg}"
        print(line, flush=True)
        with open(self.path, "a") as fh:
            fh.write(line + "\n")


def parse_duration(text):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smh])", text)
    if not m:
        raise ValueError(f"bad duration {text!r} (use e.g. 13m, 90s, 1.5h)")
    return float(m.group(1)) * {"s": 1, "m": 60, "h": 3600}[m.group(2)]


def parse_queue(path):
    runs = []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), 1):
        tokens = shlex.split(raw, comments=True)
        if not tokens:
            continue
        name, rest = tokens[0], tokens[1:]
        est = None
        if rest and rest[0].startswith("est="):
            est = parse_duration(rest[0][4:])
            rest = rest[1:]
        if "--out-dir" in rest:
            sys.exit(f"{path}:{lineno}: --out-dir is set by the queue")
        if "--seed" not in rest:
            sys.exit(f"{path}:{lineno}: explicit --seed required")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or any(r["name"] == name for r in runs):
            sys.exit(f"{path}:{lineno}: invalid or duplicate run name {name!r}")
        runs.append({"name": name, "est": est, "args": rest})
    return runs


def config_key(args):
    """train.py arguments without --seed (and --out-dir): runs with the same key cost the same."""
    key, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in ("--seed", "--out-dir"):
            skip = True
            continue
        key.append(a)
    return tuple(key)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def estimate(run, out_root):
    """(seconds, source) or (None, reason)."""
    key = config_key(run["args"])
    walls = []
    for timing_file in out_root.glob("*/queue_timing.json"):
        t = read_json(timing_file)
        cfg = read_json(timing_file.parent / "config.json")
        if t and cfg and cfg.get("status") == "finished" and tuple(t.get("config_key", [])) == key \
                and t.get("train_wall_s") is not None and t.get("eval_wall_s") is not None:
            walls.append((t["train_wall_s"] + t["eval_wall_s"], timing_file.parent.name))
    if walls:
        sec, src = max(walls)
        return sec, f"max of {len(walls)} finished run(s) with the same configuration (longest: {src})"
    if run["est"] is not None:
        return run["est"], "est= in the queue file"
    return None, "no finished run with the same configuration and no est= in the queue file"


def next_cutoff(hhmm, start):
    if "T" in hhmm:  # absolute local date-time YYYY-MM-DDTHH:MM
        cut = datetime.fromisoformat(hhmm).astimezone()  # naive -> local time zone (DST-aware)
        if cut <= start:
            raise SystemExit(f"cutoff {hhmm} is not after the start {stamp(start)}")
        return cut
    h, m = (int(x) for x in hhmm.split(":"))
    cut = start.replace(hour=h, minute=m, second=0, microsecond=0)
    return cut if cut > start else cut + timedelta(days=1)


def git_dirty():
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    return out.strip()


def machine_state():
    parts = []
    for f in sorted(Path("/sys/class/power_supply").glob("*/online")):
        parts.append(f"{f.parent.name}_online={f.read_text().strip()}")
    try:
        parts.append("platform_profile=" + Path("/sys/firmware/acpi/platform_profile").read_text().strip())
    except OSError:
        pass
    mem = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()
           if l.startswith(("MemAvailable", "SwapTotal", "SwapFree"))}
    parts.append(f"mem_available_mb={mem['MemAvailable'] // 1024} "
                 f"swap_used_mb={(mem['SwapTotal'] - mem['SwapFree']) // 1024}")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,pstate,memory.used",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
        parts.append("gpu=" + gpu.strip().replace(" ", ""))
    except (OSError, subprocess.SubprocessError):
        parts.append("gpu=n/a")
    return " ".join(parts)


def run_child(cmd, log_path):
    """Run cmd (own session, so a terminal Ctrl-C reaches only the queue). Returns (exit code, seconds)."""
    global _child
    t0 = time.perf_counter()
    with open(log_path, "w") as fh:
        _child = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        rc = _child.wait()
    _child = None
    return rc, time.perf_counter() - t0


def mark_interrupted(run_dir, reason):
    cfg_path = run_dir / "config.json"
    cfg = read_json(cfg_path)
    if cfg is None:
        return
    cfg.update({"status": "interrupted", "interrupted_reason": reason, "interrupted_local": stamp()})
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")


def on_signal(signum, _frame):
    global _stop_signal
    _stop_signal = signal.Signals(signum).name
    if _child is not None and _child.poll() is None:
        _child.terminate()


def eval_splits(run_dir):
    """Validation splits a finished run is evaluated on: val, plus val_seen for grouped-holdout runs."""
    cfg = read_json(run_dir / "config.json") or {}
    return ["val", "val_seen"] if cfg.get("split", {}).get("split_mode") == "grouped-holdout" else ["val"]


def evaluate(name, run_dir, out_root, log):
    total = 0.0
    for split in eval_splits(run_dir):
        log(f"{name}: evaluation on {split} started")
        rc, sec = run_child([PY, "scripts/evaluate.py", "--run-dir", str(run_dir), "--split", split],
                            out_root / f"{name}.eval_{split}.log")
        if rc != 0:
            log(f"{name}: evaluation FAILED (exit {rc}, {sec:.0f} s), see {name}.eval_{split}.log")
            return None
        res = read_json(run_dir / f"eval_{split}.json")
        m = res["means"]
        log(f"{name}: evaluation finished in {sec:.0f} s; {split} rel_all {100 * m['rel_all']:.4f} % "
            f"rel_p {100 * m['rel_p']:.4f} rel_vx {100 * m['rel_vx']:.4f} rel_vy {100 * m['rel_vy']:.4f} "
            f"({res['n_samples']} samples, split {res.get('split_mode')})")
        total += sec
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("queue_file", type=Path)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--cutoff", default="07:00", help="HH:MM local time, next occurrence after start; or YYYY-MM-DDTHH:MM (absolute, local time)")
    args = ap.parse_args()

    runs = parse_queue(args.queue_file)
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log = Log(out_root / "queue.log")
    start = now_local()
    cutoff = next_cutoff(args.cutoff, start)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    log(f"=== queue start: {args.queue_file} ({len(runs)} runs), out root {out_root}, cutoff {stamp(cutoff)}, "
        f"commit {commit}, python {PY}, pid {os.getpid()}, TMPDIR {os.environ.get('TMPDIR')}")
    if git_dirty():
        log("working tree is dirty; commit first. Queue not started.")
        sys.exit(1)

    for run in runs:
        name = run["name"]
        run_dir = out_root / name
        stop_file = out_root / "STOP"
        if stop_file.exists():
            stop_file.rename(out_root / f"STOP.consumed-{utc_tag()}")
            log(f"STOP file found before {name}: queue stopped (file renamed to STOP.consumed-*)")
            break
        if _stop_signal:
            break

        need_train = True
        if run_dir.exists():
            cfg = read_json(run_dir / "config.json")
            status = cfg.get("status") if cfg else None
            if status == "finished" and all((run_dir / f"eval_{s}.json").exists() for s in eval_splits(run_dir)):
                log(f"{name}: already finished and evaluated - skipped")
                continue
            if status == "finished":
                log(f"{name}: training finished but not evaluated - evaluation only")
                need_train = False
            else:
                tag = utc_tag()
                aside = out_root / f"{name}.interrupted-{tag}"
                run_dir.rename(aside)
                for suffix in (".log", ".eval_val.log", ".eval_val_seen.log"):
                    old = out_root / f"{name}{suffix}"
                    if old.exists():
                        old.rename(out_root / f"{name}.interrupted-{tag}{suffix}")
                log(f"{name}: found with status {status!r} - treated as interrupted, moved to {aside.name}, "
                    f"redone from scratch")

        if need_train:
            est, source = estimate(run, out_root)
            now = now_local()
            if est is None:
                log(f"{name}: not started ({source}); queue stopped")
                break
            if now + timedelta(seconds=est) > cutoff:
                log(f"{name}: not started - estimated {est / 60:.1f} min ({source}) would end at "
                    f"{stamp(now + timedelta(seconds=est))}, after the cutoff {stamp(cutoff)}; queue stopped")
                break
            log(f"{name}: training started: train.py {shlex.join(run['args'])}; estimated {est / 60:.1f} min "
                f"({source}), expected end {stamp(now + timedelta(seconds=est))}")
            log(f"{name}: machine state {machine_state()}")
            cmd = [PY, "scripts/train.py", *run["args"], "--out-dir", str(run_dir)]
            rc, train_sec = run_child(cmd, out_root / f"{name}.log")
            if rc != 0 or _stop_signal:
                reason = f"exit code {rc}" + (f" after {_stop_signal} to the queue" if _stop_signal else "")
                mark_interrupted(run_dir, reason)
                log(f"{name}: training INTERRUPTED ({reason}) after {train_sec:.0f} s; status set to interrupted")
                if _stop_signal:
                    break
                continue
            log(f"{name}: training finished in {train_sec:.0f} s ({train_sec / 60:.1f} min)")
        else:
            t = read_json(run_dir / "queue_timing.json") or {}
            train_sec = t.get("train_wall_s")

        eval_sec = evaluate(name, run_dir, out_root, log)
        (run_dir / "queue_timing.json").write_text(json.dumps({
            "config_key": list(config_key(run["args"])),
            "train_wall_s": train_sec,
            "eval_wall_s": eval_sec,
            "finished_local": stamp(),
        }, indent=2) + "\n")
        if _stop_signal:
            break

    if _stop_signal:
        log(f"queue stopped by {_stop_signal}")
    log(f"=== queue end; machine state {machine_state()}")


if __name__ == "__main__":
    main()
