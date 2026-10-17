#!/usr/bin/env bash
# Launch the night-8b queue (Saturday 2026-10-17, early morning; docs/preregistration-night8b.md: Conv64, seeds 0-4,
# grouped split, validation only) as nights 1-8a: detached tmux
# session, sleep / idle / lid switch inhibited, cutoff 12:00 (2026-10-17), out root ~/runs/night8b.
# Refuses to start if the working tree is dirty (tracked or untracked changes) or if the tmux session exists. Prints the
# expected end (nominal and worst case, from the queue file) before starting.
#
# Usage: scripts/launch_night8b.sh
# Overridable: QUEUE (queues/night8b.txt), OUT_ROOT (~/runs/night8b), CUTOFF (12:00), SESSION (night8b)
# Monitor: tail -f ~/runs/night8b/queue.log    Stop after the current run: touch ~/runs/night8b/STOP
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
QUEUE="${QUEUE:-queues/night8b.txt}"
OUT_ROOT="${OUT_ROOT:-$HOME/runs/night8b}"
CUTOFF="${CUTOFF:-12:00}"
SESSION="${SESSION:-night8b}"
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

# expected end: sum of `# nominal=<N>m` (nominal) and of est= (worst case) over the runs of the queue
"$PY" - "$QUEUE" "$CUTOFF" <<'PYEOF'
import re, sys
from datetime import datetime, timedelta
sys.path.insert(0, "scripts")
from run_queue import next_cutoff, parse_queue
path, cutoff = sys.argv[1], sys.argv[2]
runs = parse_queue(path)
nominal = sum(float(m.group(1)) for m in re.finditer(r"#\s*nominal=([\d.]+)m", open(path).read()))
worst = sum(r["est"] for r in runs) / 60
now = datetime.now().astimezone()
cut = next_cutoff(cutoff, now)
fmt = lambda m: (now + timedelta(minutes=m)).strftime("%a %H:%M")
print(f"Queue {path}: {len(runs)} runs; expected end {fmt(nominal)} (nominal, {nominal / 60:.1f} h), "
      f"worst case {fmt(worst)} ({worst / 60:.1f} h); cutoff {cut.strftime('%a %H:%M')}")
if now + timedelta(minutes=worst) > cut:
    print("WARNING: the worst case ends after the cutoff: the last runs may not start (check the launch rule).")
PYEOF

mkdir -p "$HOME/.cache/pip-tmp" "$OUT_ROOT"
tmux new -d -s "$SESSION" "export TMPDIR=\$HOME/.cache/pip-tmp; systemd-inhibit --what=sleep:idle:handle-lid-switch \
--why=${SESSION}-queue '$PY' scripts/run_queue.py '$QUEUE' --out-root '$OUT_ROOT' --cutoff $CUTOFF"
sleep 5
echo "Launched: tmux session '$SESSION', commit $(git rev-parse --short HEAD), queue $QUEUE, out root $OUT_ROOT, cutoff $CUTOFF"
echo "--- last lines of $OUT_ROOT/queue.log:"
tail -n 3 "$OUT_ROOT/queue.log" 2>/dev/null || echo "(no queue.log yet)"
