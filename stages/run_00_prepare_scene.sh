#!/usr/bin/env bash
# 用途：阶段 00 启动入口；整理输入图片或 2:1 全景图，并写入场景元信息。
set -euo pipefail
STAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$STAGE_DIR/../scripts/env.sh"
source "$HY2_SCRIPT_ROOT/activate_env.sh"
python "$HY2_PY_ROOT/prepare_scene.py" --run-root "$HY2_RUN_ROOT" "$@"
