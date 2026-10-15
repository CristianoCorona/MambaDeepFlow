#!/usr/bin/env bash
# Single test-set session of 2026-10-16 (docs/lab-notebook.md, 2026-10-15; pre-registrations d46b279). ONE pass:
#   1. scripts/evaluate.py --split test (batch 1, one run at a time) on the NEW runs only: MDv64 s0-7 (night5 s0-2,
#      night6 s3-7) and the 9 grouped-holdout runs (night6 GH_{B,MD8,MDv8-16-32}_s0-2); B s0-7 were evaluated on test
#      in the first pass (ca067a4) and are only read from the cache;
#   2. scripts/final_analysis.py --confirm-mdv64 on test (decision: CONFIRMED / NOT CONFIRMED);
#   3. scripts/seen_unseen.py --split test --allow-test (calibration of the validations, descriptive).
# Runs missing or not finished are skipped (pre-registered: available seeds, no further runs).
#
# Refuses: without the argument --go; on a dirty working tree; if any of the new runs already has eval_test.json
# (single pass). Everything is logged to ~/runs/test-session-night6/session.log; results go to docs/results/ and
# docs/figures/ and are committed right after, without re-runs.
#
# Usage: scripts/test_session_night6.sh --go
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
R="$HOME/runs"
LOGDIR="$R/test-session-night6"
cd "$REPO"

[ "${1:-}" = "--go" ] || { echo "Single pre-registered test pass: run with --go" >&2; exit 1; }
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is dirty; commit first:" >&2; git status --short >&2; exit 1
fi
PY="$(conda run -n mdf python -c 'import sys; print(sys.executable)')"
export TMPDIR="$HOME/.cache/pip-tmp"
mkdir -p "$TMPDIR" "$LOGDIR"

NEW_RUNS=()
for d in "$R"/night5/MDv64_s{0,1,2} "$R"/night6/MDv64_s{3,4,5,6,7} "$R"/night6/GH_{B,MD8,MDv8-16-32}_s{0,1,2}; do
    if [ -f "$d/config.json" ] && grep -q '"status": "finished"' "$d/config.json"; then
        NEW_RUNS+=("$d")
    else
        echo "skipped (missing or not finished): $d" | tee -a "$LOGDIR/session.log"
    fi
done
for d in "${NEW_RUNS[@]}"; do
    if [ -e "$d/eval_test.json" ]; then echo "ERROR: $d already has eval_test.json (single pass)" >&2; exit 1; fi
done

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOGDIR/session.log"; }
log "=== test session start, commit $(git rev-parse HEAD), ${#NEW_RUNS[@]} new runs"
for d in "${NEW_RUNS[@]}"; do
    log "evaluate.py --split test $d"
    "$PY" scripts/evaluate.py --run-dir "$d" --split test > "$LOGDIR/$(basename "$(dirname "$d")")_$(basename "$d").eval_test.log" 2>&1
    log "  done: $(grep -A1 'Relative error % (original' "$LOGDIR/$(basename "$(dirname "$d")")_$(basename "$d").eval_test.log" | tail -1)"
done

log "final_analysis.py --confirm-mdv64 --split test"
"$PY" scripts/final_analysis.py --config "B=$R/night1/ep133_s*" --config "B=$R/night3/B_s*" \
    --config "MDv64=$R/night5/MDv64_s*" --config "MDv64=$R/night6/MDv64_s*" --seeds 0-7 \
    --split test --allow-test --confirm-mdv64 \
    --out-dir docs/results/test-confirm-mdv64 --fig-dir docs/figures/test-confirm-mdv64 2>&1 | tee "$LOGDIR/confirm.log"

log "seen_unseen.py --split test --allow-test"
"$PY" scripts/seen_unseen.py --config "B=$R/night6/GH_B_s*" --config "MD8=$R/night6/GH_MD8_s*" \
    --config "MDv8-16-32=$R/night6/GH_MDv8-16-32_s*" --split test --allow-test \
    --out-dir docs/results/seen-unseen-test 2>&1 | tee "$LOGDIR/seen_unseen.log"
log "=== test session end"
