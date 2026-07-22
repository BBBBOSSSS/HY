#!/usr/bin/env bash
# Download the two HF-hosted HY2 model groups directly, without a proxy.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_VENV="/root/autodl-tmp/_bundle_deps/download-tools"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/hf-model-download-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'rc=$?; echo "[$(date -Is)] HF model download stopped (exit $rc). Log: $LOG_FILE"; exit "$rc"' EXIT

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="*"
export no_proxy="*"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_ENDPOINT="https://hf-mirror.com"

if [[ ! -x "$TOOLS_VENV/bin/python" ]]; then
  /root/miniconda3/bin/python -m venv "$TOOLS_VENV"
fi
"$TOOLS_VENV/bin/python" -m pip install -U huggingface_hub
export PATH="$TOOLS_VENV/bin:$PATH"

bash "$ROOT/模型下载/model_download.sh" \
  --only worldstereo,uni3c \
  --hf-endpoint "$HF_ENDPOINT" \
  --skip-preflight
