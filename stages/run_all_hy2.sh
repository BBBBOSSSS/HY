#!/usr/bin/env bash
# 用途：底层全链路入口；顺序执行阶段 00 到阶段 06，但不负责启动或关闭 vLLM。
STAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$STAGE_DIR/../scripts/env.sh"
source "$HY2_STAGE_ROOT/env.sh"
NAME=""; IMAGE=""; PANO=""; MODE="vlm-full"; NPROC="${NPROC_PER_NODE:-1}"; NFRAME="${HY2_NFRAME:-21}"; STEPS=12000; PROMPT=""; SCENE_TYPE="${HY2_SCENE_TYPE:-outdoor}"; FSDP=0
ALIGN_NFRAME="${HY2_ALIGN_NFRAME:-8}"; MAX_REFERENCE="${HY2_MAX_REFERENCE:-8}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --panorama) PANO="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --nframe) NFRAME="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --scene-type) SCENE_TYPE="$2"; shift 2 ;;
    --align-nframe) ALIGN_NFRAME="$2"; shift 2 ;;
    --max-reference) MAX_REFERENCE="$2"; shift 2 ;;
    --fsdp) FSDP=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$NAME" ]] || { echo "--name required" >&2; exit 2; }
if [[ -n "$PANO" ]]; then
  bash "$HY2_STAGE_ROOT/run_00_prepare_scene.sh" --name "$NAME" --panorama "$PANO" --scene-type "$SCENE_TYPE"
elif [[ -n "$IMAGE" ]]; then
  bash "$HY2_STAGE_ROOT/run_00_prepare_scene.sh" --name "$NAME" --image "$IMAGE" --scene-type "$SCENE_TYPE"
  if [[ -n "$PROMPT" ]]; then bash "$HY2_STAGE_ROOT/run_01_panorama.sh" --name "$NAME" --prompt "$PROMPT"; else bash "$HY2_STAGE_ROOT/run_01_panorama.sh" --name "$NAME"; fi
else
  echo "Provide --image or --panorama" >&2; exit 2
fi
bash "$HY2_STAGE_ROOT/run_02_worldnav.sh" --name "$NAME" --mode "$MODE" --nframe "$NFRAME"
bash "$HY2_STAGE_ROOT/run_03_traj_render.sh" --name "$NAME" --mode "$MODE" --nproc "$NPROC" --expected-nframe "$NFRAME"
ws_args=(--name "$NAME" --nproc "$NPROC" --align-nframe "$ALIGN_NFRAME" --max-reference "$MAX_REFERENCE")
[[ "$FSDP" == 1 ]] && ws_args+=(--fsdp)
bash "$HY2_STAGE_ROOT/run_04_worldstereo.sh" "${ws_args[@]}"
bash "$HY2_STAGE_ROOT/run_05_gs_data.sh" --name "$NAME" --nproc "$NPROC"
bash "$HY2_STAGE_ROOT/run_06_train_gs.sh" --name "$NAME" --steps "$STEPS" --nproc "$NPROC"
echo "DONE: $HY2_RUN_ROOT/$NAME"
