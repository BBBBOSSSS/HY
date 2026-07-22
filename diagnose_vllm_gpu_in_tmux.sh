#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="$ROOT/logs/vllm-gpu-diagnose-20260721.log"
source "$ROOT/scripts/env.sh"
export HY2_VLLM_VENV="/root/autodl-tmp/_bundle_deps/conda-envs/vllm_qwen"
export CUDA_VISIBLE_DEVICES=0
export CONDA_PREFIX="$HY2_VLLM_VENV"
export PATH="$HY2_VLLM_VENV/bin:$PATH"
export LD_LIBRARY_PATH="$HY2_VLLM_VENV/lib/python3.11/site-packages/nvidia/cu13/lib:$HY2_VLLM_VENV/lib"
unset PYTHONPATH

{
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  "$HY2_VLLM_VENV/bin/python" - <<'PY'
import sys
import torch
print("python", sys.executable)
print("torch", torch.__version__, "torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
PY
} 2>&1 | tee -a "$LOG"

exec bash
