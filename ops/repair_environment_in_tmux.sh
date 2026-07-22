#!/usr/bin/env bash
# Repair the incomplete no-GPU HY2 setup while keeping the cu128 runtime stack.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/environment-repair-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'rc=$?; echo "[$(date -Is)] environment repair stopped (exit $rc). Log: $LOG_FILE"; exit "$rc"' EXIT

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="*"
export no_proxy="*"

source "$ROOT/env.sh"
source "$ROOT/activate_env.sh"

echo "== 1/4 Align optional CuPy with CUDA 12 =="
"$HY2_VENV/bin/python" -m pip uninstall -y cupy-cuda13x cuda-pathfinder || true
"$HY2_VENV/bin/python" -m pip install cupy-cuda12x==14.1.1

echo "== 2/4 Install MoGe source package =="
MOGE_SRC="$HY2_BUNDLE_ROOT/upstream/moge_src/MoGe-main"
if [[ ! -d "$MOGE_SRC/.git" ]]; then
  mkdir -p "$(dirname "$MOGE_SRC")"
  rm -rf "$MOGE_SRC"
  for attempt in 1 2 3; do
    if git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/microsoft/MoGe.git "$MOGE_SRC"; then
      break
    fi
    rm -rf "$MOGE_SRC"
    if [[ "$attempt" == 3 ]]; then
      echo "MoGe source clone failed after $attempt attempts." >&2
      exit 1
    fi
    echo "MoGe clone attempt $attempt failed; retrying in 10 seconds..." >&2
    sleep 10
  done
fi
"$HY2_VENV/bin/python" -m pip install -e "$MOGE_SRC"

echo "== 3/4 Create the vLLM environment =="
bash "$ROOT/conda环境配置/setup_conda_env.sh" --skip-gsplat --skip-preflight

echo "== 4/4 Verify environment packages =="
source "$ROOT/env.sh"
source "$ROOT/activate_env.sh"
python - <<'PY'
import moge, torch
print('MoGe:', moge.__file__)
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
PY
"$HY2_BUNDLE_ROOT/conda-envs/vllm_qwen/bin/python" - <<'PY'
import vllm
print('vLLM:', vllm.__version__)
PY
