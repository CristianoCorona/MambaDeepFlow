#!/usr/bin/env bash
# Launch the night-10 queue (Sunday 2026-10-18 night; docs/preregistration-final.md, commit 45dc422): MDv64-local
# seeds 5-7, grouped split, 133 epochs, same arguments as night 8a. As nights 1-9: detached tmux session, sleep / idle /
# lid switch inhibited, out root ~/runs/night10, cutoff given as the absolute date-time 2026-10-19T08:00 (no run
# started after it). A failed run is repeated identically once by running this launcher again (finished runs are
# skipped; a later launch needs a new CUTOFF=...).
# Refuses to start if the working tree is dirty (tracked or untracked changes), if the tmux session exists, or if HEAD
# does not descend from the pre-registration commit. Prints the expected end (nominal and worst case, from the queue
# file) before starting.
#
# Usage: scripts/launch_night10.sh
# Overridable: QUEUE (queues/night10.txt), OUT_ROOT (~/runs/night10), CUTOFF (2026-10-19T08:00), SESSION (night10)
# Monitor: tail -f ~/runs/night10/queue.log    Stop after the current run: touch ~/runs/night10/STOP
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
QUEUE="${QUEUE:-queues/night10.txt}"
OUT_ROOT="${OUT_ROOT:-$HOME/runs/night10}"
CUTOFF="${CUTOFF:-2026-10-19T08:00}"
SESSION="${SESSION:-night10}"
PREREG=45dc4220519faeccdbabc2bfac39338078adeb20  # docs/preregistration-final.md
cd "$REPO"

if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is dirty; commit (or remove) these changes first, then relaunch:" >&2
    git status --short >&2
    exit 1
fi
if ! git merge-base --is-ancestor "$PREREG" HEAD; then
    echo "ERROR: HEAD does not descend from the pre-registration commit $PREREG" >&2
    exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists (queue already running?). Check: tmux attach -t $SESSION" >&2
    exit 1
fi
[ -f "$QUEUE" ] || { echo "ERROR: queue file $QUEUE not found" >&2; exit 1; }
PY="$(conda run -n mdf python -c 'import sys; print(sys.executable)')"

# expected end: sum of `# nominal=<N>m` (nominal) and of est= (worst case) over the runs of the queue
"$PY" - "$QUEUE" "$CUTOFF" <<'PYEOF'
import re, sys
from datetime import datetime, timedelta
sys.path.insert(0, "scripts")
from run_queue import next_cutoff, parse_queue
path, cutoff = sys.argv[1], sys.argv[2]
runs = parse_queue(path)
lines = [l for l in open(path) if l.strip() and not l.lstrip().startswith("#")]
nom = [float(re.search(r"#\s*nominal=([\d.]+)m", l).group(1)) for l in lines]
assert len(nom) == len(runs), "every run line needs a '# nominal=<N>m' comment"
nominal, worst = sum(nom), sum(r["est"] for r in runs) / 60
now = datetime.now().astimezone()
cut = next_cutoff(cutoff, now)
fmt = lambda m: (now + timedelta(minutes=m)).strftime("%a %H:%M")
print(f"Queue {path}: {len(runs)} runs; expected end {fmt(nominal)} (nominal, {nominal / 60:.1f} h), "
      f"worst case {fmt(worst)} ({worst / 60:.1f} h); cutoff {cut.strftime('%a %d/%m %H:%M')}")
if now + timedelta(minutes=worst) > cut:
    print("WARNING: the worst case ends after the cutoff: the last runs may not start.")
PYEOF

mkdir -p "$HOME/.cache/pip-tmp" "$OUT_ROOT"
tmux new -d -s "$SESSION" "export TMPDIR=\$HOME/.cache/pip-tmp; systemd-inhibit --what=sleep:idle:handle-lid-switch \
--why=${SESSION}-queue '$PY' scripts/run_queue.py '$QUEUE' --out-root '$OUT_ROOT' --cutoff $CUTOFF"
sleep 5
echo "Launched: tmux session '$SESSION', commit $(git rev-parse --short HEAD), queue $QUEUE, out root $OUT_ROOT, cutoff $CUTOFF"
echo "--- last lines of $OUT_ROOT/queue.log:"
tail -n 3 "$OUT_ROOT/queue.log" 2>/dev/null || echo "(no queue.log yet)"
