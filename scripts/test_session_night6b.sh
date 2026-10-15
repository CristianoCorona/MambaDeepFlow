#!/usr/bin/env bash
# Single test-set session for night 6b (Friday 2026-10-16; pre-registration docs/preregistration-night6b.md, commit
# a986ca1). ONE pass:
#   1. scripts/evaluate.py --split test (batch 1, one run at a time) on the night-6b runs only: DRv11 / GH_B / TR s0-4
#      and GH_MD8 / GH_MDv8-16-32 s0-2 (~/runs/night6b);
#   2. scripts/final_analysis.py --confirm-dr on test: DRv11 - B-GH on rel_all (IMPROVEMENT / WORSENING DEMONSTRATED or
#      NO DEMONSTRABLE DIFFERENCE) and the TR control (TR - B-GH, DRv11 - TR, descriptive) -> docs/results/test-confirm-dr;
#   3. scripts/seen_unseen.py --split test --allow-test: calibration of the validations over all GH configurations
#      (B, MD8, MDv8-16-32, DRv11, TR; descriptive) -> docs/results/seen-unseen-test.
# Runs missing or not finished are skipped (pre-registered: available seeds, no further runs).
#
# Refuses: without the argument --go; on a dirty working tree; if any of the runs already has eval_test.json (single
# pass). Everything is logged to ~/runs/test-session-night6b/session.log; results go to docs/results/ and docs/figures/
# and are committed right after, without re-runs.
#
# Usage: scripts/test_session_night6b.sh --go
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
R="$HOME/runs"
N="$R/night6b"
LOGDIR="$R/test-session-night6b"
cd "$REPO"

[ "${1:-}" = "--go" ] || { echo "Single pre-registered test pass: run with --go" >&2; exit 1; }
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is dirty; commit first:" >&2; git status --short >&2; exit 1
fi
PY="$(conda run -n mdf python -c 'import sys; print(sys.executable)')"
export TMPDIR="$HOME/.cache/pip-tmp"
mkdir -p "$TMPDIR" "$LOGDIR"

RUNS=()
for d in "$N"/{DRv11,GH_B,TR}_s{0,1,2,3,4} "$N"/GH_{MD8,MDv8-16-32}_s{0,1,2}; do
    if [ -f "$d/config.json" ] && grep -q '"status": "finished"' "$d/config.json"; then
        RUNS+=("$d")
    else
        echo "skipped (missing or not finished): $d" | tee -a "$LOGDIR/session.log"
    fi
done
for d in "${RUNS[@]}"; do
    if [ -e "$d/eval_test.json" ]; then echo "ERROR: $d already has eval_test.json (single pass)" >&2; exit 1; fi
done

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOGDIR/session.log"; }
log "=== test session start, commit $(git rev-parse HEAD), ${#RUNS[@]} runs"
for d in "${RUNS[@]}"; do
    log "evaluate.py --split test $d"
    "$PY" scripts/evaluate.py --run-dir "$d" --split test > "$LOGDIR/$(basename "$d").eval_test.log" 2>&1
    log "  done: $(grep -A1 'Relative error % (original' "$LOGDIR/$(basename "$d").eval_test.log" | tail -1)"
done

log "final_analysis.py --confirm-dr --split test"
"$PY" scripts/final_analysis.py --config "B-GH=$N/GH_B_s*" --config "DRv11=$N/DRv11_s*" --config "TR=$N/TR_s*" \
    --seeds 0-4 --split test --allow-test --confirm-dr \
    --out-dir docs/results/test-confirm-dr --fig-dir docs/figures/test-confirm-dr 2>&1 | tee "$LOGDIR/confirm.log"

log "seen_unseen.py --split test --allow-test"
"$PY" scripts/seen_unseen.py --config "B=$N/GH_B_s*" --config "MD8=$N/GH_MD8_s*" \
    --config "MDv8-16-32=$N/GH_MDv8-16-32_s*" --config "DRv11=$N/DRv11_s*" --config "TR=$N/TR_s*" \
    --split test --allow-test --out-dir docs/results/seen-unseen-test 2>&1 | tee "$LOGDIR/seen_unseen.log"
log "=== test session end"
