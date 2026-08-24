#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/WMA_HW4
WMA_DIR="$ROOT/unifolm-world-model-action"
TOOLS_DIR="$ROOT/asc-wma-performance-homework"
ENV_DIR="$ROOT/wma-env"
RESULTS_ROOT="$ROOT/experiment_results"

if [[ $# -ne 1 ]]; then
  echo "usage: bash $0 <experiment_label>"
  echo "example: bash $0 baseline_a800_01"
  exit 2
fi

LABEL="$1"
if [[ ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "experiment_label may contain only letters, digits, dot, underscore and hyphen"
  exit 2
fi

AMP_DTYPE="${WMA_AMP_DTYPE:-none}"
INFERENCE_MODE="${WMA_INFERENCE_MODE:-0}"
N_ITER="${WMA_N_ITER:-11}"
if [[ ! "$AMP_DTYPE" =~ ^(none|fp16|bf16)$ ]]; then
  echo "WMA_AMP_DTYPE must be one of: none, fp16, bf16"
  exit 2
fi
if [[ ! "$N_ITER" =~ ^[1-9][0-9]*$ ]]; then
  echo "WMA_N_ITER must be a positive integer"
  exit 2
fi
if [[ ! "$INFERENCE_MODE" =~ ^(0|1)$ ]]; then
  echo "WMA_INFERENCE_MODE must be 0 or 1"
  exit 2
fi

RESULT_DIR="$RESULTS_ROOT/$LABEL"
if [[ -e "$RESULT_DIR" ]]; then
  echo "refusing to mix or overwrite an existing experiment: $RESULT_DIR"
  exit 1
fi

mkdir -p "$RESULT_DIR/output"
export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$ROOT/hf-cache"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1

cleanup_monitor() {
  if [[ -n "${MONITOR_PID:-}" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup_monitor EXIT INT TERM

bash "$TOOLS_DIR/scripts/collect_colab_env.sh" "$RESULT_DIR/environment.txt"
{
  echo "cgroup_memory_limit_bytes=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo unknown)"
  echo "model_commit=$(git -C "$WMA_DIR" rev-parse HEAD)"
  echo "homework_commit=$(git -C "$TOOLS_DIR" rev-parse HEAD)"
  echo "amp_dtype=$AMP_DTYPE"
  echo "inference_mode=$INFERENCE_MODE"
  echo "n_iter=$N_ITER"
} > "$RESULT_DIR/reproducibility.txt"

sha256sum \
  "$WMA_DIR/ckpts/unifolm_wma_dual.ckpt" \
  "$WMA_DIR/unitree_g1_pack_camera/case1/unitree_g1_pack_camera_case1.mp4" \
  "$WMA_DIR/unitree_g1_pack_camera/case1/world_model_interaction_prompts/unitree_g1_pack_camera.csv" \
  "$WMA_DIR/unitree_g1_pack_camera/case1/world_model_interaction_prompts/images/unitree_g1_pack_camera/0.png" \
  "$WMA_DIR/unitree_g1_pack_camera/case1/world_model_interaction_prompts/transitions/unitree_g1_pack_camera/0.h5" \
  > "$RESULT_DIR/input_sha256.txt"

QUERY="timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
nvidia-smi --query-gpu="$QUERY" --format=csv -l 1 > "$RESULT_DIR/gpu_monitor.csv" &
MONITOR_PID=$!

MODEL_ARGS=(
  scripts/evaluation/world_model_interaction.py
  --seed 123
  --ckpt_path ckpts/unifolm_wma_dual.ckpt
  --config configs/inference/world_model_interaction.yaml
  --savedir "$RESULT_DIR/output"
  --bs 1
  --height 320
  --width 512
  --unconditional_guidance_scale 1.0
  --ddim_steps 50
  --ddim_eta 1.0
  --prompt_dir unitree_g1_pack_camera/case1/world_model_interaction_prompts
  --dataset unitree_g1_pack_camera
  --video_length 16
  --frame_stride 6
  --n_action_steps 16
  --exe_steps 16
  --n_iter "$N_ITER"
  --timestep_spacing uniform_trailing
  --guidance_rescale 0.7
  --perframe_ae
)

if [[ "$AMP_DTYPE" != "none" ]]; then
  MODEL_ARGS+=(--amp_dtype "$AMP_DTYPE")
fi
if [[ "$INFERENCE_MODE" == "1" ]]; then
  MODEL_ARGS+=(--inference_mode)
fi

cd "$WMA_DIR"
bash "$TOOLS_DIR/scripts/time_command.sh" "$RESULT_DIR/run.log" \
  "$ENV_DIR/bin/python" "${MODEL_ARGS[@]}"

cleanup_monitor
unset MONITOR_PID

python "$TOOLS_DIR/scripts/summarize_gpu_monitor.py" \
  "$RESULT_DIR/gpu_monitor.csv" | tee "$RESULT_DIR/gpu_summary.txt"

PRED_VIDEO="$RESULT_DIR/output/inference/0_full_fs6.mp4"
python "$ROOT/psnr_score_for_challenge.py" \
  --gt_video unitree_g1_pack_camera/case1/unitree_g1_pack_camera_case1.mp4 \
  --pred_video "$PRED_VIDEO" \
  --output_file "$RESULT_DIR/psnr_result.json" \
  | tee "$RESULT_DIR/psnr.log"

ffprobe -v error \
  -show_entries format=duration,size:stream=codec_name,width,height,avg_frame_rate,nb_frames \
  -of default=noprint_wrappers=1 "$PRED_VIDEO" \
  | tee "$RESULT_DIR/video_info.txt"

echo "experiment complete: $RESULT_DIR"
