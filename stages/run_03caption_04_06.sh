#!/usr/bin/env bash
# 用途：断点续跑入口；重新描述轨迹视频，然后继续执行阶段 04 到阶段 06。
set -euo pipefail
STAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$STAGE_DIR/../scripts/env.sh"
source "$HY2_SCRIPT_ROOT/activate_env.sh"

NAME=""
NPROC="${NPROC_PER_NODE:-1}"
RENDER_NPROC="${HY2_RENDER_NPROC:-}"
STEPS=12000
VLLM_SESSION="${HY2_VLLM_SESSION:-hy2_vllm}"
KILL_VLLM=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --render-nproc) RENDER_NPROC="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --vllm-session) VLLM_SESSION="$2"; shift 2 ;;
    --keep-vllm) KILL_VLLM=0; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$NAME" ]] || { echo "--name required" >&2; exit 2; }
RENDER_NPROC="${RENDER_NPROC:-$NPROC}"

RUN_DIR="$HY2_RUN_ROOT/$NAME"
SCENE_DIR="$RUN_DIR/scene"
[[ -d "$SCENE_DIR/render_results" ]] || { echo "Missing render_results: $SCENE_DIR/render_results" >&2; exit 1; }
mkdir -p "$RUN_DIR/logs"

exec > >(tee -a "$RUN_DIR/logs/stage03_06_stable.tmux.log") 2>&1

echo "[run_03caption_04_06] name=$NAME render_nproc=$RENDER_NPROC later_nproc=$NPROC steps=$STEPS"
echo "[run_03caption_04_06] scene=$SCENE_DIR"

CAMERA_COUNT=$(find "$SCENE_DIR/render_results" -type f -name camera.json | wc -l)
VIDEO_COUNT=$(find "$SCENE_DIR/render_results" -type f -name render.mp4 | wc -l)
CAPTION_COUNT=$(find "$SCENE_DIR/render_results" -type f -name traj_caption.json | wc -l)
if [[ "$CAMERA_COUNT" -eq 0 ]]; then
  echo "No camera.json files found under $SCENE_DIR/render_results" >&2
  exit 1
elif [[ "$VIDEO_COUNT" -lt "$CAMERA_COUNT" ]]; then
  echo "[run_03caption_04_06] incomplete renders: videos=$VIDEO_COUNT cameras=$CAMERA_COUNT; running Stage03"
  bash "$HY2_STAGE_ROOT/run_03_traj_render.sh" --name "$NAME" --mode vlm-full --nproc "$RENDER_NPROC" --skip-exist
elif [[ "$CAPTION_COUNT" -lt "$VIDEO_COUNT" ]]; then
  echo "[run_03caption_04_06] incomplete captions: captions=$CAPTION_COUNT videos=$VIDEO_COUNT; refreshing VLM captions"
  python "$HY2_PY_ROOT/recaption_traj_videos.py" \
    --scene-dir "$SCENE_DIR" \
    --llm-addr 127.0.0.1 \
    --llm-port "$HY2_LLM_PORT" \
    --llm-name "$HY2_LLM_NAME" \
    --workers "${HY2_CAPTION_WORKERS:-1}" \
    2>&1 | tee "$RUN_DIR/logs/03_recaption_stable.log"
else
  echo "[run_03caption_04_06] Stage03 complete: cameras=$CAMERA_COUNT videos=$VIDEO_COUNT captions=$CAPTION_COUNT; skipping rendering/captioning"
fi

if [[ "$KILL_VLLM" == 1 ]]; then
  tmux kill-session -t "$VLLM_SESSION" 2>/dev/null || true
  sleep 5
fi

bash "$HY2_STAGE_ROOT/run_04_worldstereo.sh" --name "$NAME" --nproc "$NPROC" --skip-exist
bash "$HY2_STAGE_ROOT/run_05_gs_data.sh" --name "$NAME" --nproc "$NPROC"
bash "$HY2_STAGE_ROOT/run_06_train_gs.sh" --name "$NAME" --steps "$STEPS" --nproc "$NPROC"

echo "[run_03caption_04_06] DONE: $RUN_DIR"
