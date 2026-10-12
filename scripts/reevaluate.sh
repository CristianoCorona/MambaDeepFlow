#!/usr/bin/env bash
# Re-run scripts/evaluate.py --split val on existing run directories (validation only), keeping the previous
# outputs as eval_val.v1.json / eval_val_samples.v1.csv (never overwritten once present).
# Usage: scripts/reevaluate.sh RUN_DIR...
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$(conda run -n mdf python -c 'import sys; print(sys.executable)')}"
cd "$REPO"
for d in "$@"; do
    [ -f "$d/eval_val.v1.json" ] || cp -p "$d/eval_val.json" "$d/eval_val.v1.json"
    [ -f "$d/eval_val_samples.v1.csv" ] || cp -p "$d/eval_val_samples.csv" "$d/eval_val_samples.v1.csv"
    "$PY" scripts/evaluate.py --run-dir "$d" --split val > "$d/reevaluate.log" 2>&1
    "$PY" - "$d" <<'PYEOF'
import json, sys
d = sys.argv[1]
old = json.load(open(f"{d}/eval_val.v1.json"))["means"]
new = json.load(open(f"{d}/eval_val.json"))
diff = max(abs(old[k] - new["means"][k]) for k in old)
s = new["secondary"]
print(f"{d}: max |primary new - old| = {diff:.2e}; split {new['split_mode']}; agg rel_all {100*s['agg_rel_all']:.4f} "
      f"median rel_all {100*s['median_rel_all']:.4f} agg rel_p {100*s['agg_rel_p']:.4f} median rel_p {100*s['median_rel_p']:.4f}")
PYEOF
done
