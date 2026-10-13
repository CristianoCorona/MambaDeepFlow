#!/usr/bin/env bash
# Launch the night-3 queue (docs/lab-notebook.md, "Night 3: pre-registered seed extension") as nights 1-2:
# detached tmux session, sleep / idle / lid switch inhibited, cutoff 08:30, out root ~/runs/night3.
# Refuses to start if the working tree is dirty (tracked or untracked changes) or if the tmux session exists.
#
# Usage: scripts/launch_night3.sh
# Overridable: QUEUE (queues/night3.txt), OUT_ROOT (~/runs/night3), CUTOFF (08:30), SESSION (night3)
# Monitor: tail -f ~/runs/night3/queue.log    Stop after the current run: touch ~/runs/night3/STOP
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
QUEUE="${QUEUE:-queues/night3.txt}"
OUT_ROOT="${OUT_ROOT:-$HOME/runs/night3}"
CUTOFF="${CUTOFF:-08:30}"
SESSION="${SESSION:-night3}"
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
