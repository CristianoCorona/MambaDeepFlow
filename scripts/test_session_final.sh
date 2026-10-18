#!/usr/bin/env bash
# Single test-set session of the final pass (docs/preregistration-final.md, commit 45dc422): the fifth and LAST test
# pass of the project. ONE pass:
#   1. scripts/evaluate.py --split test (batch 1, one run at a time) on MDv64-local s0-7 only (night8a s0-4, night10
#      s5-7); B s0-7 (pass 1, ca067a4) and MDv64 s0-7 (pass 4, 817c558) were evaluated on test before and are only read
#      from the explore cache;
#   2. scripts/final_analysis.py --confirm-final on test: MDv64-local - B on rel_vy, CONFIRMED / NOT CONFIRMED (printed
#      first, between banners) + descriptive MDv64-local - B and MDv64-local - MDv64 -> docs/results/test-final,
#      docs/figures/test-final.
#
# Refuses: without --go (or --dry-run); on a dirty working tree; if HEAD does not descend from the pre-registration
# commit; if any MDv64-local run already has eval_test.json or a test entry in the explore cache (single pass); if a
# B / MDv64 reference run lacks eval_test.json; if an MDv64-local run is missing or not finished without being
# declared with --declare-missing (pre-registered: a failed run is repeated once, otherwise the test uses the seeds
# available and the missing ones are declared in the notebook before the pass), or if a declared seed is in fact
# present.
# Everything is logged to ~/runs/test-session-final/session.log; results are committed right after, without re-runs.
#
# --dry-run: trial mode that NEVER reads the test set. The test split is replaced by the validation split everywhere:
# evaluate.py --split val with --out-dir in a scratch directory (run directories untouched), final_analysis.py --split
# val (banner "NOT THE TEST SET"), outputs and log in $DRY_ROOT (default $HOME/runs/test-session-final-dryrun). The
# dirty-tree check is a warning in this mode only; all other checks are the same as with --go.
#
# Usage: scripts/test_session_final.sh --go [--declare-missing 5,6]
#        scripts/test_session_final.sh --dry-run [--declare-missing 5,6,7]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
R="$HOME/runs"
PREREG=45dc4220519faeccdbabc2bfac39338078adeb20  # docs/preregistration-final.md
cd "$REPO"

MODE="${1:-}"
case "$MODE" in
    --go) SPLIT=test; LOGDIR="$R/test-session-final"; OUT=docs/results/test-final; FIG=docs/figures/test-final ;;
    --dry-run) SPLIT=val; DRY_ROOT="${DRY_ROOT:-$R/test-session-final-dryrun}"; LOGDIR="$DRY_ROOT"
               OUT="$DRY_ROOT/results"; FIG="$DRY_ROOT/figures" ;;
    *) echo "Single pre-registered test pass: run with --go (or --dry-run: validation instead of test)" >&2; exit 1 ;;
esac
shift
DECLARED=""
if [ "${1:-}" = "--declare-missing" ]; then
    [ -n "${2:-}" ] || { echo "ERROR: --declare-missing needs a comma-separated list of seeds" >&2; exit 1; }
    DECLARED="$2"; shift 2
fi
[ $# -eq 0 ] || { echo "ERROR: unexpected arguments: $*" >&2; exit 1; }

if [ -n "$(git status --porcelain)" ]; then
    if [ "$MODE" = "--go" ]; then
        echo "ERROR: working tree is dirty; commit first:" >&2; git status --short >&2; exit 1
    fi
    echo "WARNING (dry run only): working tree is dirty"
fi
if ! git merge-base --is-ancestor "$PREREG" HEAD; then
    echo "ERROR: HEAD does not descend from the pre-registration commit $PREREG" >&2; exit 1
fi

# MDv64-local runs: s0-4 night 8a, s5-7 night 10; missing / unfinished ones must be exactly the declared seeds
RUNS=(); MISSING=()
for s in 0 1 2 3 4 5 6 7; do
    if [ "$s" -le 4 ]; then d="$R/night8a/MDv64-local_s$s"; else d="$R/night10/MDv64-local_s$s"; fi
    if [ -f "$d/config.json" ] && grep -q '"status": "finished"' "$d/config.json" && [ -f "$d/model.pt" ]; then
        RUNS+=("$d")
    else
        MISSING+=("$s")
    fi
done
MISSING_STR="$(IFS=,; echo "${MISSING[*]:-}")"
if [ "$MISSING_STR" != "$DECLARED" ]; then
    echo "ERROR: MDv64-local seeds missing or not finished: '${MISSING_STR}'; declared with --declare-missing: '${DECLARED}'." >&2
    echo "       They must match (declare the missing seeds in the notebook first, then pass --declare-missing)." >&2
    exit 1
fi
for d in "${RUNS[@]}"; do
    if [ -e "$d/eval_test.json" ]; then echo "ERROR: $d already has eval_test.json (single pass)" >&2; exit 1; fi
    if compgen -G "$R/explore-cache/$(basename "$(dirname "$d")")__$(basename "$d")__test__*" > /dev/null; then
        echo "ERROR: the explore cache already holds a test inference of $d (single pass)" >&2; exit 1
    fi
done
for d in "$R"/night1/ep133_s{0,1,2} "$R"/night3/B_s{3,4,5,6,7} "$R"/night5/MDv64_s{0,1,2} "$R"/night7/MDv64_s{3,4,5,6,7}; do
    [ -f "$d/eval_test.json" ] || { echo "ERROR: reference run $d has no eval_test.json" >&2; exit 1; }
done

PY="$(conda run -n mdf python -c 'import sys; print(sys.executable)')"
export TMPDIR="$HOME/.cache/pip-tmp"
mkdir -p "$TMPDIR" "$LOGDIR"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOGDIR/session.log"; }
log "=== ${MODE#--} session start (split $SPLIT), commit $(git rev-parse HEAD), ${#RUNS[@]} MDv64-local runs" \
    "(declared missing: ${DECLARED:-none})"
for d in "${RUNS[@]}"; do
    name="$(basename "$(dirname "$d")")_$(basename "$d")"
    if [ "$MODE" = "--go" ]; then
        log "evaluate.py --split test $d"
        "$PY" scripts/evaluate.py --run-dir "$d" --split test > "$LOGDIR/$name.eval_test.log" 2>&1
    else
        log "evaluate.py --split val $d --out-dir $LOGDIR/eval/$name (dry run)"
        "$PY" scripts/evaluate.py --run-dir "$d" --split val --out-dir "$LOGDIR/eval/$name" \
            > "$LOGDIR/$name.eval_val.log" 2>&1
    fi
    log "  done: $(grep -A1 'Relative error % (original' "$LOGDIR/$name.eval_$SPLIT.log" | tail -1)"
done

ALLOW=(); [ "$SPLIT" = "test" ] && ALLOW=(--allow-test)
log "final_analysis.py --confirm-final --split $SPLIT -> $OUT"
"$PY" scripts/final_analysis.py --config "B=$R/night1/ep133_s*" --config "B=$R/night3/B_s*" \
    --config "MDv64=$R/night5/MDv64_s*" --config "MDv64=$R/night7/MDv64_s*" \
    --config "MDv64-local=$R/night8a/MDv64-local_s*" --config "MDv64-local=$R/night10/MDv64-local_s*" \
    --seeds 0-7 --split "$SPLIT" "${ALLOW[@]}" --confirm-final \
    --out-dir "$OUT" --fig-dir "$FIG" 2>&1 | tee "$LOGDIR/confirm.log"
log "=== ${MODE#--} session end"
