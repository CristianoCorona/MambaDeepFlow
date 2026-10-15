#!/usr/bin/env bash
# Launch the night-6b queue (docs/preregistration-night6b.md, commit a986ca1: DRv11 / B-GH / TR seeds 0-4, then the
# grouped-holdout runs MD8 / MDv8-16-32 seeds 0-2 moved from night 6) as nights 1-6: detached tmux session, sleep /
# idle / lid switch inhibited, cutoff 08:30, out root ~/runs/night6b. Each run is evaluated on val and val_seen by
# scripts/run_queue.py (grouped-holdout runs); STOP file as in run_queue.py.
# Refuses to start if the working tree is dirty (tracked or untracked changes) or if the tmux session exists.
#
# Usage: scripts/launch_night6b.sh
# Overridable: QUEUE (queues/night6b.txt), OUT_ROOT (~/runs/night6b), CUTOFF (08:30), SESSION (night6b)
# Monitor: tail -f ~/runs/night6b/queue.log    Stop after the current run: touch ~/runs/night6b/STOP
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
QUEUE="${QUEUE:-queues/night6b.txt}"
OUT_ROOT="${OUT_ROOT:-$HOME/runs/night6b}"
CUTOFF="${CUTOFF:-08:30}"
SESSION="${SESSION:-night6b}"
cd "$REPO"

if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is dirty; commit (or remove) these changes first, then relaunch:" >&2
    git status --short >&2
    exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists (queue already running?). Check: tmux attach -t $SESSION" >&2
    exit 1
fi
[ -f "$QUEUE" ] || { echo "ERROR: queue file $QUEUE not found" >&2; exit 1; }
PY="$(conda run -n mdf python -c 'import sys; print(sys.executable)')"
mkdir -p "$HOME/.cache/pip-tmp" "$OUT_ROOT"

tmux new -d -s "$SESSION" "export TMPDIR=\$HOME/.cache/pip-tmp; systemd-inhibit --what=sleep:idle:handle-lid-switch \
--why=${SESSION}-queue '$PY' scripts/run_queue.py '$QUEUE' --out-root '$OUT_ROOT' --cutoff $CUTOFF"
sleep 5
echo "Launched: tmux session '$SESSION', commit $(git rev-parse --short HEAD), queue $QUEUE, out root $OUT_ROOT, cutoff $CUTOFF"
echo "--- last lines of $OUT_ROOT/queue.log:"
tail -n 3 "$OUT_ROOT/queue.log" 2>/dev/null || echo "(no queue.log yet)"
