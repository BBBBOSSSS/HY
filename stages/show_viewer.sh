#!/usr/bin/env bash
# 用途：查看器入口；从最新 checkpoint 打开官方 HY-World GS viewer。
set -euo pipefail
STAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$STAGE_DIR/../scripts/env.sh"
source "$HY2_SCRIPT_ROOT/activate_env.sh"
if [[ -n "${HY2_CUDA13_ROOT:-}" && -x "$HY2_CUDA13_ROOT/bin/nvcc" ]]; then
  export CUDA_HOME="$HY2_CUDA13_ROOT"
  export CUDA_PATH="$HY2_CUDA13_ROOT"
  export CUDACXX="$HY2_CUDA13_ROOT/bin/nvcc"
  export PATH="$HY2_CUDA13_ROOT/bin:$PATH"
else
  export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
  export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
  export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
fi
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/root/autodl-tmp/.cache/torch_extensions}"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
NAME=""; PORT=7007; GPU_ID=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$NAME" ]] || { echo "--name required" >&2; exit 2; }
RUN_DIR="$HY2_RUN_ROOT/$NAME"
CKPT=$(ls -1 "$RUN_DIR"/gs_result/ckpts/ckpt_*_rank*.pt 2>/dev/null | sort -V | tail -n 1 || true)
[[ -n "$CKPT" ]] || { echo "no ckpt found under $RUN_DIR/gs_result/ckpts" >&2; exit 2; }
python "$HY2_PY_ROOT/require_cuda.py" --stage "HY-World GS viewer"
cd "$HY2_WORLDGEN_ROOT"
python show_gs.py --port "$PORT" --gpu_id "$GPU_ID" --ckpt "$CKPT"
