#!/usr/bin/env bash
# 用途：阶段 02 启动入口；运行 WorldNav 深度、视角切分、导航区域和相机轨迹规划。
set -euo pipefail
STAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$STAGE_DIR/../scripts/env.sh"
source "$HY2_SCRIPT_ROOT/activate_env.sh"
NAME=""; MODE="vlm-full"; NFRAME=21; SPLIT_RES=720; FOV_X=""; FOV_Y=""; ENABLE_NAV=1; EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --nframe) NFRAME="$2"; shift 2 ;;
    --splitted-resolution) SPLIT_RES="$2"; shift 2 ;;
    --fov-x) FOV_X="$2"; shift 2 ;;
    --fov-y) FOV_Y="$2"; shift 2 ;;
    --regular-only) ENABLE_NAV=0; shift ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
[[ -n "$NAME" ]] || { echo "--name required" >&2; exit 2; }
SCENE_DIR="$HY2_RUN_ROOT/$NAME/scene"
[[ -f "$SCENE_DIR/panorama.png" ]] || { echo "missing $SCENE_DIR/panorama.png" >&2; exit 2; }
python "$HY2_PY_ROOT/require_cuda.py" --stage "stage02 WorldNav trajectory generation"
cd "$HY2_WORLDGEN_ROOT"
args=(--target_path "$SCENE_DIR" --nframe "$NFRAME" --splitted_resolution "$SPLIT_RES" --llm_addr "$HY2_LLM_ADDR" --llm_port "$HY2_LLM_PORT" --llm_name "$HY2_LLM_NAME")
[[ -n "$FOV_X" ]] && args+=(--fov_x "$FOV_X")
[[ -n "$FOV_Y" ]] && args+=(--fov_y "$FOV_Y")
if [[ "$ENABLE_NAV" == 1 ]]; then
  args+=(--apply_nav_traj --apply_up_route --apply_recon_iteration)
fi
if [[ "$MODE" == "vlm-full" ]]; then
  export WORLDGEN_SKIP_VLLM=0
  args+=(--force_vlm)
else
  export WORLDGEN_SKIP_VLLM=1
fi
{
  echo "[run_02_worldnav] mode=$MODE nframe=$NFRAME split_res=$SPLIT_RES enable_nav=$ENABLE_NAV"
  python traj_generate.py "${args[@]}" "${EXTRA[@]}"
  python "$HY2_PY_ROOT/check_camera_json_counts.py" --scene-dir "$SCENE_DIR" --expected-count "$NFRAME"
} 2>&1 | tee "$HY2_RUN_ROOT/$NAME/logs/02_worldnav.log"
