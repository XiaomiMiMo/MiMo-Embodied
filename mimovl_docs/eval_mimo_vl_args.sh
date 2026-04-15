#!/bin/bash

MODEL_PATH=$1
TASK=$2

# Resolve script root first
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Use a relative default output dir under the repo
EVAL_RESULTS_DIR="${3:-${SCRIPT_DIR}/eval_results}"
DISABLE_THINKING_USER="${4:-false}"

MODEL_NAME=$(basename "${MODEL_PATH}")

export PYTHONPATH="${SCRIPT_DIR}/patches:${PYTHONPATH}"

# Environment variables for ML (with fallbacks for local/single-node execution)
export NNODES=${MLP_WORKER_NUM:-1}
export NODE_RANK=${MLP_ROLE_INDEX:-0}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-127.0.0.1}
export MASTER_PORT=${MLP_WORKER_0_PORT:-29500}

# Automatically detect GPUs if not provided by cluster environment
if [ -n "$MLP_WORKER_GPU" ]; then
    export NPROC_PER_NODE="$MLP_WORKER_GPU"
elif command -v nvidia-smi &> /dev/null; then
    export NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
else
    export NPROC_PER_NODE=1
fi

export GPT_EVAL_NUM_RETRIES=5
export GPT_EVAL_NUM_SECONDS_TO_SLEEP=10
export GPT_EVAL_RAISE_AFTER_MAX_RETRIES=0
export GPT_EVAL_TIMEOUT=30

# Configuration for image and video preprocessing
PATCH_SIZE=28
IMAGE_MIN_TOKENS=0
IMAGE_MAX_TOKENS=128

VIDEO_MIN_TOKENS=0
VIDEO_MAX_TOKENS=1024
VIDEO_TOTAL_MAX_TOKENS=2048
VIDEO_FPS=1
VIDEO_MIN_FRAMES=0
VIDEO_MAX_FRAMES=32
VIDEO_NFRAMES=0

IMAGE_MIN_PIXELS=$((IMAGE_MIN_TOKENS * PATCH_SIZE * PATCH_SIZE))
IMAGE_MAX_PIXELS=$((IMAGE_MAX_TOKENS * PATCH_SIZE * PATCH_SIZE))
VIDEO_MIN_PIXELS=$((VIDEO_MIN_TOKENS * PATCH_SIZE * PATCH_SIZE))
VIDEO_MAX_PIXELS=$((VIDEO_MAX_TOKENS * PATCH_SIZE * PATCH_SIZE))
VIDEO_TOTAL_MAX_PIXELS=$((VIDEO_TOTAL_MAX_TOKENS * PATCH_SIZE * PATCH_SIZE))

export QWEN_RESIZE_MAX_PIXELS="$IMAGE_MAX_PIXELS"

kwargs=""
if [ "$IMAGE_MIN_PIXELS" -gt 0 ]; then
    kwargs="${kwargs},image_min_pixels=${IMAGE_MIN_PIXELS}"
fi
if [ "$IMAGE_MAX_PIXELS" -gt 0 ]; then
    kwargs="${kwargs},image_max_pixels=${IMAGE_MAX_PIXELS}"
fi
if [ "$VIDEO_MIN_PIXELS" -gt 0 ]; then
    kwargs="${kwargs},video_min_pixels=${VIDEO_MIN_PIXELS}"
fi
if [ "$VIDEO_MAX_PIXELS" -gt 0 ]; then
    kwargs="${kwargs},video_max_pixels=${VIDEO_MAX_PIXELS}"
fi
if [ "$VIDEO_TOTAL_MAX_PIXELS" -gt 0 ]; then
    kwargs="${kwargs},video_total_max_pixels=${VIDEO_TOTAL_MAX_PIXELS}"
fi
if [ "$VIDEO_FPS" -gt 0 ]; then
    kwargs="${kwargs},video_fps=${VIDEO_FPS}"
fi
if [ "$VIDEO_MIN_FRAMES" -gt 0 ]; then
    kwargs="${kwargs},video_min_frames=${VIDEO_MIN_FRAMES}"
fi
if [ "$VIDEO_MAX_FRAMES" -gt 0 ]; then
    kwargs="${kwargs},video_max_frames=${VIDEO_MAX_FRAMES}"
fi
if [ "$VIDEO_NFRAMES" -gt 0 ]; then
    kwargs="${kwargs},video_nframes=${VIDEO_NFRAMES}"
fi
if [ "$DISABLE_THINKING_USER" = "true" ]; then
    kwargs="${kwargs},disable_thinking_user=true"
fi

NUM_PROCESS=$((NNODES * NPROC_PER_NODE))

# Dynamically build accelerate arguments
ACCELERATE_ARGS="--num_processes=${NUM_PROCESS} --machine_rank=${NODE_RANK} --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} --num_machines=${NNODES}"
if [ "$NUM_PROCESS" -gt 1 ]; then
    ACCELERATE_ARGS="--multi_gpu ${ACCELERATE_ARGS}"
fi

mkdir -p "${EVAL_RESULTS_DIR}/${MODEL_NAME}"

echo "Evaluating ${TASK} with kwargs: ${kwargs}"
python3 -m accelerate.commands.launch ${ACCELERATE_ARGS} \
    -m lmms_eval \
    --model mivllm \
    --model_args model_version=${MODEL_PATH},max_model_len=8192,gpu_memory_utilization=0.85,max_num_seqs=16,max_images=120,max_videos=10,max_audios=0,dtype=bfloat16${kwargs} \
    --tasks "${TASK}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${MODEL_NAME}" \
    --output_path "${EVAL_RESULTS_DIR}/${MODEL_NAME}"