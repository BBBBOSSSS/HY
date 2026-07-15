#!/usr/bin/env bash
# 用途：服务脚本；在 tmux 中启动 Qwen3-VL 的 vLLM OpenAI 兼容服务。
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"

if [[ -d "${HY2_VLLM_VENV:-$HY2_BUNDLE_ROOT/conda-envs/vllm_qwen}" ]]; then
  export HY2_VLLM_VENV="${HY2_VLLM_VENV:-$HY2_BUNDLE_ROOT/conda-envs/vllm_qwen}"
else
  export HY2_VLLM_VENV="${HY2_VLLM_VENV:-/root/autodl-tmp/conda-envs/vllm_qwen}"
fi
if [[ ! -x "$HY2_VLLM_VENV/bin/vllm" ]]; then
  echo "vLLM executable not found: $HY2_VLLM_VENV/bin/vllm" >&2
  exit 1
fi
PYTHONPATH= "$HY2_VLLM_VENV/bin/python" "$HY2_PY_ROOT/require_cuda.py" --stage "vLLM Qwen3-VL server"

SESSION="${1:-hy2_vllm}"
CMD="source '$HY2_SCRIPT_ROOT/env.sh'; \
export HY2_VLLM_VENV='$HY2_VLLM_VENV'; \
export CONDA_PREFIX=\"\$HY2_VLLM_VENV\"; \
unset PYTHONPATH; \
export PATH=\"\$HY2_VLLM_VENV/bin:\$PATH\"; \
export LD_LIBRARY_PATH=\"\$HY2_VLLM_VENV/lib/python3.11/site-packages/nvidia/cu13/lib:\$HY2_VLLM_VENV/lib\"; \
export VLLM_USE_FLASHINFER_SAMPLER='$VLLM_USE_FLASHINFER_SAMPLER'; \
\"\$HY2_VLLM_VENV/bin/vllm\" serve '$QWEN3_VL_MODEL_PATH' \
  --host 0.0.0.0 \
  --port '$HY2_LLM_PORT' \
  --served-model-name '$HY2_LLM_NAME' \
  --tensor-parallel-size '$HY2_VLLM_TP_SIZE' \
  --gpu-memory-utilization '$HY2_VLLM_GPU_UTIL' \
  --max-model-len '$HY2_VLLM_MAX_MODEL_LEN' \
  --limit-mm-per-prompt '{\"image\": '$HY2_VLLM_MM_LIMIT_IMAGE'}' \
  --trust-remote-code \
  \$HY2_VLLM_EXTRA_ARGS"
tmux new -s "$SESSION" -d "$CMD"
echo "tmux attach -t $SESSION"
