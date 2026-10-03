#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/WMA20
MODEL_DIR="$ROOT/unifolm-world-model-action"
CASES_DIR="$ROOT/ASC26-Embodied-World-Model-Optimization"
PYTHON="$ROOT/wma-env/bin/python"
RESULT_DIR="${1:?usage: bash score_persistent_20.sh /absolute/result/directory}"

if [[ "$RESULT_DIR" != /* || ! -f "$RESULT_DIR/summary.csv" ]]; then
  echo "Expected an absolute result directory with summary.csv" >&2
  exit 2
fi

shopt -s nullglob
printf 'scenario,case,n_iter,frame_stride,psnr,frames\n' > "$RESULT_DIR/quality.csv"

while IFS=, read -r scenario case_name n_iter frame_stride elapsed; do
  if [[ "$scenario" == scenario ]]; then
    continue
  fi
  case_result="$RESULT_DIR/$scenario/$case_name"
  references=("$MODEL_DIR/$scenario/$case_name"/*.mp4)
  videos=("$case_result/output/inference/"*_full_fs"${frame_stride}".mp4)
  if [[ ${#references[@]} -ne 1 || ${#videos[@]} -ne 1 || ! -s "${videos[0]}" ]]; then
    echo "Missing or ambiguous video: $scenario/$case_name" >&2
    exit 1
  fi

  "$PYTHON" "$CASES_DIR/psnr_score_for_challenge.py" \
    --gt_video "${references[0]}" --pred_video "${videos[0]}" \
    --output_file "$case_result/psnr_result.json" > "$case_result/psnr.log" 2>&1
  ffprobe -v error \
    -show_entries format=duration:stream=codec_name,width,height,avg_frame_rate,nb_frames \
    -of default=noprint_wrappers=1 "${videos[0]}" > "$case_result/video_info.txt"
  frames=$(sed -n 's/^nb_frames=//p' "$case_result/video_info.txt" | head -1)
  if [[ "$frames" != "$((n_iter * 16))" ]]; then
    echo "Frame count mismatch: $scenario/$case_name" >&2
    exit 1
  fi
  psnr=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["psnr"])' "$case_result/psnr_result.json")
  printf '%s,%s,%s,%s,%s,%s\n' \
    "$scenario" "$case_name" "$n_iter" "$frame_stride" "$psnr" "$frames" \
    >> "$RESULT_DIR/quality.csv"
  echo "SCORED $scenario/$case_name PSNR=${psnr}dB frames=$frames"
done < "$RESULT_DIR/summary.csv"

if [[ $(wc -l < "$RESULT_DIR/quality.csv") -ne 21 ]]; then
  echo "Expected quality scores for all 20 cases" >&2
  exit 1
fi
