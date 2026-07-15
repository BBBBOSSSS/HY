#!/usr/bin/env bash
# 用途：环境激活脚本；激活 HY2 Python 环境并设置 CUDA 运行变量。
set -euo pipefail

if [[ -f "$HY2_VENV/bin/activate" ]]; then
  source "$HY2_VENV/bin/activate"
elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # cuda-nvcc conda activation scripts assume this exists when shell nounset is on
  export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
  export _CONDA_PYTHON_SYSCONFIGDATA_NAME_USED="${_CONDA_PYTHON_SYSCONFIGDATA_NAME_USED:-}"
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "$HY2_VENV"
else
  export PATH="$HY2_VENV/bin:$PATH"
fi

if [[ -n "${HY2_CUDA13_ROOT:-}" && -d "$HY2_CUDA13_ROOT" ]]; then
  export CUDA_HOME="${CUDA_HOME:-$HY2_CUDA13_ROOT}"
  export CUDA_PATH="${CUDA_PATH:-$HY2_CUDA13_ROOT}"
  export PATH="$HY2_CUDA13_ROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$HY2_CUDA13_ROOT/lib:$HY2_VENV/lib:${LD_LIBRARY_PATH:-}"
fi
