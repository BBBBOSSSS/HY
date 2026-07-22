#!/usr/bin/env bash
# Reproducible, detach-safe setup entrypoint for this checkout.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup-$(date +%Y%m%d-%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1
trap 'rc=$?; echo "[$(date -Is)] setup stopped (exit $rc). Log: $LOG_FILE"; exit "$rc"' EXIT

echo "[$(date -Is)] HY2 setup started"
echo "Pipeline root: $ROOT"
echo "Log: $LOG_FILE"

echo "== 1/4 Prepare upstream source =="
bash "$ROOT/源码准备/setup_source_code.sh"

echo "== 2/4 Create HY2 and vLLM environments (CPU/no-GPU setup) =="
# Models are deliberately downloaded in the next step, so the environment
# install must not run the model-dependent preflight yet.
bash "$ROOT/conda环境配置/setup_conda_env.sh" --skip-gsplat --skip-preflight

echo "== 3/4 Download all required model weights =="
source "$ROOT/env.sh"
source "$ROOT/activate_env.sh"
bash "$ROOT/模型下载/model_download.sh"

echo "== 4/4 Final no-GPU preflight =="
source "$ROOT/env.sh"
source "$ROOT/activate_env.sh"
python "$ROOT/python/preflight.py"

echo "[$(date -Is)] HY2 setup completed successfully"
