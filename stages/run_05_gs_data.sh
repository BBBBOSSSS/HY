#!/usr/bin/env bash
# 用途：阶段 05 启动入口；把 WorldStereo 输出转换为 3DGS 训练数据。
set -euo pipefail
STAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$STAGE_DIR/../scripts/env.sh"
source "$HY2_SCRIPT_ROOT/activate_env.sh"
NAME=""; NPROC="${NPROC_PER_NODE:-1}"; RESULT_NAME="worldstereo-memory-dmd"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --result-name) RESULT_NAME="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$NAME" ]] || { echo "--name required" >&2; exit 2; }
SCENE_DIR="$HY2_RUN_ROOT/$NAME/scene"
cd "$HY2_WORLDGEN_ROOT"
args=(--root_path "$SCENE_DIR" --result_name "$RESULT_NAME" --save_normal --split_sky)
python "$HY2_PY_ROOT/check_frame_camera_counts.py" --scene-dir "$SCENE_DIR" --result-name "$RESULT_NAME"
python "$HY2_PY_ROOT/require_cuda.py" --stage "stage05 GS data preparation"
torchrun --standalone --nproc_per_node "$NPROC" gen_gs_data.py "${args[@]}" 2>&1 | tee "$HY2_RUN_ROOT/$NAME/logs/05_gs_data.log"
