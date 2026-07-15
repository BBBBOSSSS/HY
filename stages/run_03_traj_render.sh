#!/usr/bin/env bash
# 用途：阶段 03 启动入口；渲染规划轨迹，并生成 VLM 轨迹描述。
set -euo pipefail
STAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$STAGE_DIR/../scripts/env.sh"
source "$HY2_SCRIPT_ROOT/activate_env.sh"
NAME=""; NPROC="${NPROC_PER_NODE:-1}"; MODE="vlm-full"; EXPECTED_NFRAME="${HY2_NFRAME:-21}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --expected-nframe) EXPECTED_NFRAME="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$NAME" ]] || { echo "--name required" >&2; exit 2; }
SCENE_DIR="$HY2_RUN_ROOT/$NAME/scene"
cd "$HY2_WORLDGEN_ROOT"
if [[ "$MODE" == "vlm-full" ]]; then export WORLDGEN_SKIP_VLLM=0; else export WORLDGEN_SKIP_VLLM=1; fi
{
  echo "[run_03_traj_render] mode=$MODE expected_nframe=$EXPECTED_NFRAME nproc=$NPROC"
  python "$HY2_PY_ROOT/check_camera_json_counts.py" --scene-dir "$SCENE_DIR" --expected-count "$EXPECTED_NFRAME"
  python "$HY2_PY_ROOT/require_cuda.py" --stage "stage03 trajectory rendering"
  torchrun --standalone --nproc_per_node "$NPROC" traj_render.py --target_path "$SCENE_DIR" --llm_addr "$HY2_LLM_ADDR" --llm_port "$HY2_LLM_PORT" --llm_name "$HY2_LLM_NAME"
} 2>&1 | tee "$HY2_RUN_ROOT/$NAME/logs/03_traj_render.log"
