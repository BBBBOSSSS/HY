#!/usr/bin/env bash
# Build the GPU extension for this host, then run the full HY2 image pipeline.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="indoor_13c504b9a25f646eab0408efcd8835b9"
INPUT_IMAGE="/root/autodl-tmp/data/室内图/13c504b9a25f646eab0408efcd8835b9.jpg"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_NAME}-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'rc=$?; echo "[$(date -Is)] image test stopped (exit $rc). Log: $LOG_FILE"; exit "$rc"' EXIT

source "$ROOT/env.sh"
source "$ROOT/activate_env.sh"
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$HY2_VENV/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0
export CUDA_VISIBLE_DEVICES=0,1,2

echo "== Build gsplat for Blackwell sm_120 =="
if [[ ! -f "$HY2_GSPLAT_ROOT/gsplat/csrc.so" ]]; then
  cd "$HY2_GSPLAT_ROOT"
  python -m pip install -e . --no-build-isolation
else
  echo "gsplat extension already present: $HY2_GSPLAT_ROOT/gsplat/csrc.so"
fi

echo "== Start full indoor image test =="
cd "$ROOT"
bash "$ROOT/run_full_vlm_hy2.sh" \
  --name "$RUN_NAME" \
  --image "$INPUT_IMAGE" \
  --scene-type indoor \
  --nproc 1 \
  --steps 12000
