#!/usr/bin/env bash
# Pre-registered analysis of night 8 (docs/preregistration-night8.md): ablations of MDv64, VALIDATION ONLY.
#   1. scripts/final_analysis.py --split val, main reading: seeds 0-4 for every configuration
#      -> docs/results/val-ablation64, docs/figures/val-ablation64
#   2. secondary reading: references B / MDv64 (and MDv8) with seeds 0-7, new variants with their seeds 0-4
#      -> docs/results/val-ablation64-8seed, docs/figures/val-ablation64-8seed
#   3. verification 5a (scripts/train_fit.py) on the main-reading configurations -> docs/results/val-ablation64/train_fit
# Missing runs (cut queue, cutoff, failures) are skipped: the seed counts are printed in summary.md.
# Never reads the test set (final_analysis refuses --split test without --allow-test; train_fit reads train / val).
#
# Usage: scripts/analyze_night8.sh
# Overridable: NIGHT8 (~/runs/night8), RESULTS (docs/results), FIGURES (docs/figures), TAG (val-ablation64)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
R="$HOME/runs"
NIGHT8="${NIGHT8:-$R/night8}"
RESULTS="${RESULTS:-docs/results}"
FIGURES="${FIGURES:-docs/figures}"
TAG="${TAG:-val-ablation64}"
cd "$REPO"
PY="$(conda run -n mdf python -c 'import sys; print(sys.executable)')"
export TMPDIR="$HOME/.cache/pip-tmp"
mkdir -p "$TMPDIR"

REFS=(--config "B=$R/night1/ep133_s*" --config "B=$R/night3/B_s*"
      --config "MDv64=$R/night5/MDv64_s*" --config "MDv64=$R/night7/MDv64_s*"
      --config "MDv8=$R/night4/MDv8_s*" --config "MDv32=$R/night5/MDv32_s*")
NEW=(--config "MDv64-local=$NIGHT8/MDv64-local_s*" --config "SA64=$NIGHT8/SA64_s*"
     --config "MDv64-dec=$NIGHT8/MDv64-dec_s*" --config "MDv16=$NIGHT8/MDv16_s*")

echo "=== main reading: seeds 0-4"
"$PY" scripts/final_analysis.py "${REFS[@]}" "${NEW[@]}" --seeds 0-4 --split val \
    --out-dir "$RESULTS/$TAG" --fig-dir "$FIGURES/$TAG"
echo "=== secondary reading: references with seeds 0-7 (new variants have seeds 0-4 only)"
"$PY" scripts/final_analysis.py "${REFS[@]}" "${NEW[@]}" --seeds 0-7 --split val \
    --out-dir "$RESULTS/$TAG-8seed" --fig-dir "$FIGURES/$TAG-8seed"
echo "=== verification 5a (train vs validation), seeds 0-4"
"$PY" scripts/train_fit.py "${REFS[@]}" "${NEW[@]}" --seeds 0-4 --out-dir "$RESULTS/$TAG/train_fit"
echo "Done: $RESULTS/$TAG{,-8seed}/summary.md, $FIGURES/$TAG/ablation64_rel_vy.pdf, $RESULTS/$TAG/train_fit"
