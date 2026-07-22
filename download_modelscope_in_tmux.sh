#!/usr/bin/env bash
# Download the ModelScope-hosted HY2 checkpoints without a proxy.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_VENV="/root/autodl-tmp/_bundle_deps/download-tools"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/modelscope-download-$(date +%Y%m%d-%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1
trap 'rc=$?; echo "[$(date -Is)] ModelScope download stopped (exit $rc). Log: $LOG_FILE"; exit "$rc"' EXIT

# ModelScope is reached directly, never through the host's local proxy.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="*"
export no_proxy="*"

if [[ ! -x "$TOOLS_VENV/bin/python" ]]; then
  /root/miniconda3/bin/python -m venv "$TOOLS_VENV"
fi
"$TOOLS_VENV/bin/python" -m pip install -U pip modelscope

export PATH="$TOOLS_VENV/bin:$PATH"
bash "$ROOT/模型下载/model_download.sh" \
  --only hy-pano-lora,qwen-image,qwen3-vl,wan-i2v,sam3,dinov2,grounding-dino,moge \
  --skip-preflight
