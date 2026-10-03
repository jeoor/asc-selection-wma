#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/WMA20
BASELINE_DIR="$ROOT/amp_baseline_20_20260926_04"
BASELINE_LOG="$ROOT/amp_baseline_20_resume_02.log"
OPT_DIR="${WMA_OPT_DIR:-$ROOT/persistent_mixed_20_20260927_01}"
TOOLS_DIR="${WMA_TOOLS_DIR:-$ROOT/optimized_eval}"
RUN_TAG="${WMA_RUN_TAG:-persistent_mixed_20}"
PYTHON="$ROOT/wma-env/bin/python"

for ((attempt = 0; attempt < 360; attempt++)); do
  if grep -q 'COMPLETE 20/20' "$BASELINE_LOG" \
    && [[ $(wc -l < "$BASELINE_DIR/summary.csv") -eq 21 ]]; then
    break
  fi
  sleep 30
done

if ! grep -q 'COMPLETE 20/20' "$BASELINE_LOG" \
  || [[ $(wc -l < "$BASELINE_DIR/summary.csv") -ne 21 ]]; then
  echo "Baseline did not complete within three hours" >&2
  exit 1
fi
if [[ -e "$OPT_DIR" ]]; then
  echo "Refusing to overwrite existing result directory: $OPT_DIR" >&2
  exit 1
fi

echo "$(date -Is) START optimized 20-case run"
export HF_HOME="$ROOT/hf-cache"
export HF_HUB_OFFLINE=1
/usr/bin/time -p -o "$ROOT/${RUN_TAG}_total_time.txt" \
  "$PYTHON" "$TOOLS_DIR/run_persistent_20.py" "$OPT_DIR" \
  > "$ROOT/${RUN_TAG}_console.log" 2>&1
cp "$ROOT/${RUN_TAG}_total_time.txt" "$OPT_DIR/total_time.txt"
echo "$(date -Is) START scoring optimized outputs"
bash "$TOOLS_DIR/score_persistent_20.sh" "$OPT_DIR" \
  > "$ROOT/${RUN_TAG}_score.log" 2>&1
"$PYTHON" "$TOOLS_DIR/assess_20.py" "$BASELINE_DIR" "$OPT_DIR"
echo "$(date -Is) COMPLETE optimized run and assessment"
