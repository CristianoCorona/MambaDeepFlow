#!/usr/bin/env bash
# Launch the night-9 queue (Saturday 2026-10-17 afternoon): the ladder of one-ingredient ablations of MDv64-local
# (docs/preregistration-night9.md, seeds 0-4) followed by SA64 seeds 0-4 (docs/preregistration-night8.md, amendment of
# 2026-10-17 "SA64 in the same queue as the night-9 ladder"); grouped split, validation only. As nights 1-8b: detached
# tmux session, sleep / idle / lid switch inhibited, cutoff 13:00 of Sunday 2026-10-18 given as the absolute date-time
# 2026-10-18T13:00 (a plain 13:00 would mean Saturday 13:00 when launched on Saturday morning), out root ~/runs/night9.
# Runs not done by the cutoff are completed the following night by running this launcher again with a new cutoff
# (e.g. CUTOFF=2026-10-19T07:00; same queue and out root: finished runs are skipped).
# Refuses to start if the working tree is dirty (tracked or untracked changes) or if the tmux session exists. Prints the
# expected end of the whole queue and of the ladder (runs before SA64), nominal and worst case, from the queue file,
# before starting.
#
# Usage: scripts/launch_night9.sh
# Overridable: QUEUE (queues/night9.txt), OUT_ROOT (~/runs/night9), CUTOFF (2026-10-18T13:00), SESSION (night9)
# Monitor: tail -f ~/runs/night9/queue.log    Stop after the current run: touch ~/runs/night9/STOP
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
QUEUE="${QUEUE:-queues/night9.txt}"
OUT_ROOT="${OUT_ROOT:-$HOME/runs/night9}"
CUTOFF="${CUTOFF:-2026-10-18T13:00}"
SESSION="${SESSION:-night9}"
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

# expected end: sum of `# nominal=<N>m` (nominal) and of est= (worst case) over the runs of the queue; the same for the
# ladder (the runs before the first SA64 run)
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
n_ladder = next((i for i, r in enumerate(runs) if r["name"].startswith("SA64")), len(runs))
nominal, worst = sum(nom), sum(r["est"] for r in runs) / 60
nominal_l, worst_l = sum(nom[:n_ladder]), sum(r["est"] for r in runs[:n_ladder]) / 60
now = datetime.now().astimezone()
cut = next_cutoff(cutoff, now)
fmt = lambda m: (now + timedelta(minutes=m)).strftime("%a %H:%M")
print(f"Queue {path}: {len(runs)} runs; expected end {fmt(nominal)} (nominal, {nominal / 60:.1f} h), "
      f"worst case {fmt(worst)} ({worst / 60:.1f} h); cutoff {cut.strftime('%a %H:%M')}")
print(f"Ladder ({n_ladder} runs before SA64): expected end {fmt(nominal_l)} (nominal, {nominal_l / 60:.1f} h), "
      f"worst case {fmt(worst_l)} ({worst_l / 60:.1f} h)")
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
