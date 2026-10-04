#!/usr/bin/env bash

# Continue the interrupted reload-only diagnostic without rerunning its first
# ten completed cases. This is a segmented ablation, not a full end-to-end run.
set -euo pipefail

ROOT=/root/autodl-tmp/WMA20
EVAL="$ROOT/ablation_eval_20261004"
PYTHON="$ROOT/wma-env/bin/python"
RUNNER="$EVAL/run_ablation_20.py"
FIRST="$EVAL/full_reload_async/summary.csv"
TAIL="$EVAL/reload_async_tail10"
REFERENCE="$ROOT/unifolm-world-model-action/unitree_z1_dual_arm_stackbox_v2/case1/unitree_z1_dual_arm_stackbox_v2_case1.mp4"
PSNR="$ROOT/ASC26-Embodied-World-Model-Optimization/psnr_score_for_challenge.py"

export HF_HOME="$ROOT/hf-cache"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com

test -f "$FIRST"
test ! -e "$TAIL"
mapfile -t remaining < <("$PYTHON" - "$FIRST" "$ROOT/ASC26-Embodied-World-Model-Optimization" <<'PY'
import csv
from pathlib import Path
import sys

completed = {
    f'{row["scenario"]}/{row["case"]}'
    for row in csv.DictReader(Path(sys.argv[1]).open())
}
official = [
    f'{script.parent.parent.name}/{script.parent.name}'
    for script in sorted(Path(sys.argv[2]).glob('unitree_*/case*/run_world_model_interaction.sh'))
]
if len(completed) != 10 or len(official) != 20 or not completed.issubset(official):
    raise SystemExit('Completed/official case count mismatch')
print('\n'.join(case for case in official if case not in completed))
PY
)
test "${#remaining[@]}" -eq 10
case_args=()
for case_name in "${remaining[@]}"; do
  case_args+=(--case "$case_name")
done

echo "START_TAIL_10 ${remaining[*]}"
/usr/bin/time -p -o "$EVAL/reload_async_tail10.time.txt" \
  "$PYTHON" "$RUNNER" reload_async "$TAIL" "${case_args[@]}" \
  > "$EVAL/reload_async_tail10.log" 2>&1
cp "$EVAL/reload_async_tail10.time.txt" "$TAIL/total_time.txt"
cp "$EVAL/reload_async_tail10.log" "$TAIL/console.log"
echo COMPLETE_TAIL_10

for tf32 in enabled disabled; do
  result="$EVAL/one_case_tf32_$tf32"
  test ! -e "$result"
  extra=()
  if [[ "$tf32" == disabled ]]; then
    extra+=(--disable-tf32)
  fi
  /usr/bin/time -p -o "$EVAL/one_case_tf32_$tf32.time.txt" \
    "$PYTHON" "$RUNNER" cached_async "$result" \
    --case unitree_z1_dual_arm_stackbox_v2/case1 "${extra[@]}" \
    > "$EVAL/one_case_tf32_$tf32.log" 2>&1
  cp "$EVAL/one_case_tf32_$tf32.time.txt" "$result/total_time.txt"
  cp "$EVAL/one_case_tf32_$tf32.log" "$result/console.log"
  video="$result/unitree_z1_dual_arm_stackbox_v2/case1/output/inference/5_full_fs4.mp4"
  "$PYTHON" "$PSNR" --gt_video "$REFERENCE" --pred_video "$video" \
    --output_file "$result/psnr_result.json" > "$result/score.log" 2>&1
  ffprobe -v error \
    -show_entries format=duration:stream=codec_name,width,height,avg_frame_rate,nb_frames \
    -of default=noprint_wrappers=1 "$video" > "$result/video_info.txt"
  echo "COMPLETE_ONE_CASE_TF32_$tf32"
done

echo COMPLETE_RECOVERY_ABLATIONS
