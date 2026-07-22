#!/usr/bin/env bash
# Full DMD pipeline for the indoor-library panorama. Long-running work is
# launched by the caller in tmux; this script owns a separate vLLM tmux service.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="indoor_library_37490e79769b07beefa29081c791d0ba_dmd"
PANO="/root/autodl-tmp/data/全景图/37490e79769b07beefa29081c791d0ba.png"
VLLM_SESSION="hy2_vllm_library_dmd"
LOG="$ROOT/logs/full-dmd-library-$(date +%Y%m%d-%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1
source "$ROOT/scripts/env.sh"
source "$ROOT/scripts/activate_env.sh"

export HY2_PIPELINE_GPU_MODE=single
source "$ROOT/scripts/configure_gpu_topology.sh"
export HY2_WORLDSTEREO_MODEL_TYPE="worldstereo-memory-dmd"
export HY2_CLEAN_PLY=0
export HY2_USE_MASK_GAUSSIAN=1
export HY2_USE_ANCHOR_PROTECTION=1

wait_for_vllm() {
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if python - "$HY2_LLM_ADDR" "$HY2_LLM_PORT" <<'PY'
import json
import sys
import urllib.request

host, port = sys.argv[1], int(sys.argv[2])
try:
    with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception:
    raise SystemExit(1)
if not payload.get("data"):
    raise SystemExit(1)
PY
    then
      echo "VLLM_READY ${HY2_LLM_ADDR}:${HY2_LLM_PORT}"
      return 0
    fi
    if ! tmux has-session -t "$VLLM_SESSION" 2>/dev/null; then
      echo "vLLM session exited before readiness: $VLLM_SESSION" >&2
      return 1
    fi
    echo "WAITING_VLLM ${HY2_LLM_ADDR}:${HY2_LLM_PORT}"
    sleep 10
  done
  echo "Timed out waiting for vLLM after 900 seconds" >&2
  return 1
}

start_vllm() {
  tmux kill-session -t "$VLLM_SESSION" 2>/dev/null || true
  CUDA_VISIBLE_DEVICES="$HY2_VLLM_CUDA_VISIBLE_DEVICES" bash "$ROOT/scripts/start_vllm_qwen3vl.sh" "$VLLM_SESSION"
  wait_for_vllm
}

stop_vllm() {
  tmux kill-session -t "$VLLM_SESSION" 2>/dev/null || true
  sleep 10
}

echo "[$(date --iso-8601=seconds)] DMD_FULL_PIPELINE_START name=$NAME"
echo "WORLDSTEREO_MODEL_TYPE=$HY2_WORLDSTEREO_MODEL_TYPE"
python "$ROOT/python/preflight.py"

# The user requested that the pipeline only proceed after the VLM service is up.
start_vllm
CUDA_VISIBLE_DEVICES="$HY2_STAGE02_CUDA_VISIBLE_DEVICES" bash "$ROOT/stages/run_00_prepare_scene.sh" --name "$NAME" --panorama "$PANO" --scene-type indoor
CUDA_VISIBLE_DEVICES="$HY2_STAGE02_CUDA_VISIBLE_DEVICES" bash "$ROOT/stages/run_02_worldnav.sh" --name "$NAME" --mode vlm-full --nframe 21

# Stage 03 captions also require vLLM; restart cleanly after WorldNav.
stop_vllm
start_vllm
CUDA_VISIBLE_DEVICES="$HY2_STAGE03_CUDA_VISIBLE_DEVICES" bash "$ROOT/stages/run_03_traj_render.sh" --name "$NAME" --mode vlm-full --nproc 1 --expected-nframe 21
stop_vllm

# DMD WorldStereo is kept on one GPU to avoid this host's Blackwell NCCL fault.
CUDA_VISIBLE_DEVICES="$HY2_STAGE04_CUDA_VISIBLE_DEVICES" bash "$ROOT/stages/run_04_worldstereo.sh" --name "$NAME" --nproc 1 --model-type worldstereo-memory-dmd --align-nframe 8 --max-reference 8

# Single-card topology keeps GS preparation and training single-GPU.
CUDA_VISIBLE_DEVICES="$HY2_STAGE05_CUDA_VISIBLE_DEVICES" bash "$ROOT/stages/run_05_gs_data.sh" --name "$NAME" --nproc "$HY2_STAGE05_NPROC" --result-name worldstereo-memory-dmd
CUDA_VISIBLE_DEVICES="$HY2_STAGE06_CUDA_VISIBLE_DEVICES" bash "$ROOT/stages/run_06_train_gs.sh" --name "$NAME" --nproc "$HY2_STAGE06_NPROC" --steps 8000

echo "[$(date --iso-8601=seconds)] DMD_FULL_PIPELINE_RC=0"
echo "DONE: $HY2_RUN_ROOT/$NAME"
