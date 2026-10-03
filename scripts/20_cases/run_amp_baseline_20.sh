#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/WMA20
MODEL_DIR="$ROOT/unifolm-world-model-action"
CASES_DIR="$ROOT/ASC26-Embodied-World-Model-Optimization"
PYTHON="$ROOT/wma-env/bin/python"
RESULT_DIR="${1:?usage: bash run_amp_baseline_20.sh /absolute/result/directory}"

if [[ "$RESULT_DIR" != /* ]]; then
  echo "Result directory must be absolute: $RESULT_DIR" >&2
  exit 2
fi

if [[ "${WMA_RESUME:-0}" == 1 ]]; then
  if [[ ! -f "$RESULT_DIR/summary.csv" ]]; then
    echo "Cannot resume without summary.csv: $RESULT_DIR" >&2
    exit 2
  fi
elif [[ -e "$RESULT_DIR" ]]; then
  echo "Result directory already exists: $RESULT_DIR" >&2
  exit 2
else
  mkdir -p "$RESULT_DIR"
fi
export HF_HOME="$ROOT/hf-cache"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=0

if [[ "${WMA_RESUME:-0}" != 1 ]]; then
  printf 'scenario,case,n_iter,frame_stride,real_seconds,psnr,frames\n' > "$RESULT_DIR/summary.csv"
  {
    date -Is
    "$PYTHON" --version
    "$PYTHON" -c 'import torch; print("torch", torch.__version__, "torch_cuda", torch.version.cuda)'
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  } > "$RESULT_DIR/environment.txt"
fi

shopt -s nullglob
scripts=("$CASES_DIR"/unitree_*/case*/run_world_model_interaction.sh)
if [[ ${#scripts[@]} -ne 20 ]]; then
  echo "Expected 20 official case scripts, found ${#scripts[@]}" >&2
  exit 1
fi

for script in "${scripts[@]}"; do
  case_name=$(basename "$(dirname "$script")")
  scenario=$(basename "$(dirname "$(dirname "$script")")")
  n_iter=$(awk '$1 == "--n_iter" {print $2}' "$script")
  frame_stride=$(awk '$1 == "--frame_stride" {print $2}' "$script")
  if [[ ! "$n_iter" =~ ^[0-9]+$ || ! "$frame_stride" =~ ^[0-9]+$ ]]; then
    echo "Cannot read n_iter/frame_stride from $script" >&2
    exit 1
  fi

  inputs=("$MODEL_DIR/$scenario/$case_name"/*.mp4)
  if [[ ${#inputs[@]} -ne 1 ]]; then
    echo "Expected one reference video for $scenario/$case_name" >&2
    exit 1
  fi

  case_result="$RESULT_DIR/$scenario/$case_name"
  if grep -q "^$scenario,$case_name," "$RESULT_DIR/summary.csv"; then
    echo "$(date -Is) SKIP  $scenario/$case_name (already complete)"
    continue
  fi
  finished_videos=("$case_result/output/inference/"*_full_fs"${frame_stride}".mp4)
  if [[ -s "$case_result/time.txt" && ${#finished_videos[@]} -eq 1 ]] \
    && [[ -s "${finished_videos[0]}" ]]; then
    echo "$(date -Is) RECOVER $scenario/$case_name (inference already finished)"
  else
    if [[ -e "$case_result" ]]; then
      interrupted="$case_result.interrupted_$(date +%Y%m%d_%H%M%S)"
      mv "$case_result" "$interrupted"
      echo "$(date -Is) PRESERVED incomplete case at $interrupted"
    fi
    mkdir -p "$case_result/output"
    command=(
    "$PYTHON" scripts/evaluation/world_model_interaction.py
    --seed 123
    --ckpt_path ckpts/unifolm_wma_dual.ckpt
    --config configs/inference/world_model_interaction.yaml
    --savedir "$case_result/output"
    --bs 1 --height 320 --width 512
    --unconditional_guidance_scale 1.0
    --ddim_steps 50 --ddim_eta 1.0
    --prompt_dir "$scenario/$case_name/world_model_interaction_prompts"
    --dataset "$scenario"
    --video_length 16 --frame_stride "$frame_stride"
    --n_action_steps 16 --exe_steps 16 --n_iter "$n_iter"
    --timestep_spacing uniform_trailing
    --guidance_rescale 0.7 --perframe_ae
    --amp_dtype fp16
    )

    printf '%q ' "${command[@]}" > "$case_result/command.txt"
    printf '\n' >> "$case_result/command.txt"
    echo "$(date -Is) START $scenario/$case_name n_iter=$n_iter stride=$frame_stride"
    (
      cd "$MODEL_DIR"
      /usr/bin/time -p -o "$case_result/time.txt" "${command[@]}" > "$case_result/run.log" 2>&1
    )
  fi

  videos=("$case_result/output/inference/"*_full_fs"${frame_stride}".mp4)
  if [[ ${#videos[@]} -ne 1 ]]; then
    echo "Expected one complete output video for $scenario/$case_name" >&2
    exit 1
  fi
  if [[ ! -s "${videos[0]}" ]]; then
    echo "Expected one complete output video for $scenario/$case_name" >&2
    exit 1
  fi
  video="${videos[0]}"
  "$PYTHON" "$CASES_DIR/psnr_score_for_challenge.py" \
    --gt_video "${inputs[0]}" --pred_video "$video" \
    --output_file "$case_result/psnr_result.json" > "$case_result/psnr.log" 2>&1

  ffprobe -v error \
    -show_entries format=duration:stream=codec_name,width,height,avg_frame_rate,nb_frames \
    -of default=noprint_wrappers=1 "$video" > "$case_result/video_info.txt"
  frames=$(sed -n 's/^nb_frames=//p' "$case_result/video_info.txt" | head -1)
  expected_frames=$((n_iter * 16))
  if [[ "$frames" != "$expected_frames" ]]; then
    echo "Frame count mismatch for $scenario/$case_name: $frames != $expected_frames" >&2
    exit 1
  fi

  real_seconds=$(awk '$1 == "real" {print $2}' "$case_result/time.txt")
  psnr=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["psnr"])' "$case_result/psnr_result.json")
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$scenario" "$case_name" "$n_iter" "$frame_stride" "$real_seconds" "$psnr" "$frames" \
    >> "$RESULT_DIR/summary.csv"
  echo "$(date -Is) DONE  $scenario/$case_name time=${real_seconds}s psnr=${psnr}dB frames=$frames"
done

echo "$(date -Is) COMPLETE 20/20"
