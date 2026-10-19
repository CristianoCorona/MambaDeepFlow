#!/usr/bin/env bash
# Launch the big-dataset queue on the cloud VM (docs/cloud-guide.md step 7; docs/preregistration-bigdata.md).
# As the night launchers: detached tmux session, run_queue.py (sequential, evaluation on validation after each run,
# STOP file, cutoff as an absolute local date-time; no run starts if its estimate would end after the cutoff).
# Then, in the same tmux session: scripts/cloud/post_bigdata.py (wall band, divergence, verification 5a; POST=0 skips
# it), and the VM powers itself off (AUTO_SHUTDOWN=1, default; `sudo shutdown -c` within 3 minutes cancels it).
#
# Refuses to start if: CUTOFF is not given; the working tree is dirty; docs/preregistration-bigdata.md is not
# committed, or HEAD does not descend from the commit that added it, or its pre-registered text changed since (dated
# amendments appended at the end are allowed, 2026-10-19); the tmux session exists;
# data/train_full is missing; the smoke test outputs are missing ($SMOKE/inspect.json without problems, $SMOKE/est.json).
# est= of each run: from $SMOKE/est.json (seconds with margin, estimate_budget.py), written into an effective copy of
# the queue ($OUT_ROOT/queue-effective.txt) whose run names and train.py arguments are checked equal to queues/bigdata.txt.
#
# Usage: CUTOFF=YYYY-MM-DDTHH:MM scripts/cloud/launch_bigdata.sh
# Overridable: QUEUE (queues/bigdata.txt), OUT_ROOT (~/runs/bigdata), SMOKE (~/runs/bigdata-smoke), SESSION (bigdata),
#              POST (1), AUTO_SHUTDOWN (1)
# Monitor: tail -f ~/runs/bigdata/queue.log   Stop after the current run: touch ~/runs/bigdata/STOP
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
QUEUE="${QUEUE:-queues/bigdata.txt}"
OUT_ROOT="${OUT_ROOT:-$HOME/runs/bigdata}"
SMOKE="${SMOKE:-$HOME/runs/bigdata-smoke}"
SESSION="${SESSION:-bigdata}"
POST="${POST:-1}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
PREREG_FILE=docs/preregistration-bigdata.md
export MDF_FULL_SUBDIRS="${MDF_FULL_SUBDIRS:-reg,shear}"  # declared fallback: reg (docs/cloud-guide.md)
cd "$REPO"
# shellcheck source=scripts/cloud/env.sh
source scripts/cloud/env.sh

[ -n "${CUTOFF:-}" ] || { echo "ERROR: give CUTOFF=YYYY-MM-DDTHH:MM (suggested by smoke_test.sh / estimate_budget.py)" >&2; exit 1; }
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is dirty; the VM must run a committed state:" >&2
    git status --short >&2
    exit 1
fi
PREREG="$(git log --diff-filter=A --format=%H -- "$PREREG_FILE" | tail -n 1)"
if [ -z "$PREREG" ]; then
    echo "ERROR: $PREREG_FILE is not committed (pre-registration before any long run)" >&2
    exit 1
fi
# the pre-registered text must be unchanged: later commits may only append dated amendments at the end of the file
ORIG_LEN=$(git show "$PREREG:$PREREG_FILE" | wc -c)
if ! git merge-base --is-ancestor "$PREREG" HEAD \
        || ! cmp -s <(git show "$PREREG:$PREREG_FILE") <(head -c "$ORIG_LEN" "$PREREG_FILE"); then
    echo "ERROR: HEAD does not descend from the pre-registration commit $PREREG, or the pre-registered text of" >&2
    echo "       $PREREG_FILE was changed (only appended amendments are allowed)" >&2
    exit 1
fi
AMENDMENTS="$(grep -c '^## Amendment' "$PREREG_FILE" || true)"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists (queue already running?). Check: tmux attach -t $SESSION" >&2
    exit 1
fi
[ -e data/train_full ] || { echo "ERROR: data/train_full missing (scripts/cloud/get_full_data.sh)" >&2; exit 1; }
[ -f "$QUEUE" ] || { echo "ERROR: queue file $QUEUE not found" >&2; exit 1; }
for f in "$SMOKE/inspect.json" "$SMOKE/est.json"; do
    [ -f "$f" ] || { echo "ERROR: $f missing: run scripts/cloud/smoke_test.sh first" >&2; exit 1; }
done
mkdir -p "$OUT_ROOT"

"$PY" - "$QUEUE" "$SMOKE" "$OUT_ROOT/queue-effective.txt" "$CUTOFF" <<'PYEOF'
import json, math, sys
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, "scripts")
from run_queue import next_cutoff, parse_queue
queue, smoke, eff, cutoff = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
if "T" not in cutoff:
    sys.exit("ERROR: CUTOFF must be an absolute date-time YYYY-MM-DDTHH:MM")
insp = json.loads((smoke / "inspect.json").read_text())
import os
if ",".join(insp["subdirs_used"]) != os.environ["MDF_FULL_SUBDIRS"]:
    sys.exit(f"ERROR: smoke test used subdirs {insp['subdirs_used']}, now MDF_FULL_SUBDIRS={os.environ['MDF_FULL_SUBDIRS']}")
if insp.get("problems"):
    sys.exit(f"ERROR: inspect_full.py reported problems: {insp['problems']}")
est = json.loads((smoke / "est.json").read_text())["seconds_est"]
runs = parse_queue(queue)
lines, out = open(queue).read().splitlines(), []
for line in lines:
    s = line.strip()
    if not s or s.startswith("#"):
        out.append(line)
        continue
    name = s.split()[0]
    label = name.rsplit("_s", 1)[0]
    if label not in est:
        sys.exit(f"ERROR: no smoke estimate for {label}")
    words = [w if not w.startswith("est=") else f"est={math.ceil(est[label] / 60)}m" for w in line.split(" ")]
    out.append(" ".join(words))
eff.write_text(f"# Effective copy of {queue}: est= from {smoke}/est.json (launch_bigdata.sh); arguments unchanged.\n"
               + "\n".join(out) + "\n")
new = parse_queue(eff)
assert [(r["name"], r["args"]) for r in new] == [(r["name"], r["args"]) for r in runs], "effective queue differs"
now = datetime.now().astimezone()
cut = next_cutoff(cutoff, now)
cum = 0.0
print(f"Queue {queue}: {len(new)} runs; cutoff {cut.strftime('%a %d/%m %H:%M')}; worst-case plan:")
for r in new:
    cum += r["est"]
    end = now + timedelta(seconds=cum)
    print(f"  {r['name']:16s} est {r['est'] / 60:6.0f} min  ends by {end.strftime('%a %d/%m %H:%M')}"
          + ("" if end <= cut else "  -> after the cutoff: will not start"))
PYEOF

POST_CMD="true"
[ "$POST" = 1 ] && POST_CMD="'$PY' scripts/cloud/post_bigdata.py --out-root '$OUT_ROOT' 2>&1 | tee -a '$OUT_ROOT/post.log'"
SHUT_CMD="echo 'AUTO_SHUTDOWN=0: the VM stays on. Power it off when done (docs/cloud-guide.md).'"
[ "$AUTO_SHUTDOWN" = 1 ] && SHUT_CMD="echo 'Powering off in 3 minutes (sudo shutdown -c to cancel)' | tee -a '$OUT_ROOT/queue.log'; sudo shutdown -h +3"
tmux new -d -s "$SESSION" "export TMPDIR='$TMPDIR' MDF_FULL_SUBDIRS='$MDF_FULL_SUBDIRS'; '$PY' scripts/run_queue.py '$OUT_ROOT/queue-effective.txt' \
--out-root '$OUT_ROOT' --cutoff $CUTOFF; $POST_CMD; $SHUT_CMD; sleep 600"
sleep 5
echo "Launched: tmux session '$SESSION', commit $(git rev-parse --short HEAD), pre-registration ${PREREG:0:7} (+$AMENDMENTS amendment(s)), queue $QUEUE, out root $OUT_ROOT, cutoff $CUTOFF, subdirs $MDF_FULL_SUBDIRS, post $POST, auto shutdown $AUTO_SHUTDOWN"
echo "--- last lines of $OUT_ROOT/queue.log:"
tail -n 3 "$OUT_ROOT/queue.log" 2>/dev/null || echo "(no queue.log yet)"
