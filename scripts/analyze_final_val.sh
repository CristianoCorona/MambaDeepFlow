#!/usr/bin/env bash
# Final pass (docs/preregistration-final.md, commit 45dc422): VALIDATION-ONLY analysis with 8 seeds, before the test
# session. Descriptive: it does not change the plan of the test pass.
#   scripts/final_analysis.py --split val --confirm-final, seeds 0-7: B (night1 s0-2 + night3 s3-7), MDv64 (night5 s0-2
#   + night7 s3-7), MDv64-local (night8a s0-4 + night10 s5-7) -> docs/results/val-final-mdv64local,
#   docs/figures/val-final-mdv64local. Same arguments as the test session apart from the split.
# Only finished runs are read; the seed counts are printed in summary.md (missing night-10 seeds show up there).
# Never reads the test set (final_analysis refuses --split test without --allow-test; this script does not pass it).
#
# Usage: scripts/analyze_final_val.sh
# Overridable: NIGHT10 (~/runs/night10), RESULTS (docs/results), FIGURES (docs/figures), TAG (val-final-mdv64local)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
R="$HOME/runs"
NIGHT10="${NIGHT10:-$R/night10}"
RESULTS="${RESULTS:-docs/results}"
FIGURES="${FIGURES:-docs/figures}"
TAG="${TAG:-val-final-mdv64local}"
cd "$REPO"
PY="$(conda run -n mdf python -c 'import sys; print(sys.executable)')"
export TMPDIR="$HOME/.cache/pip-tmp"
mkdir -p "$TMPDIR"

"$PY" scripts/final_analysis.py --config "B=$R/night1/ep133_s*" --config "B=$R/night3/B_s*" \
    --config "MDv64=$R/night5/MDv64_s*" --config "MDv64=$R/night7/MDv64_s*" \
    --config "MDv64-local=$R/night8a/MDv64-local_s*" --config "MDv64-local=$NIGHT10/MDv64-local_s*" \
    --seeds 0-7 --split val --confirm-final --out-dir "$RESULTS/$TAG" --fig-dir "$FIGURES/$TAG"
echo "Done: $RESULTS/$TAG/summary.md"
