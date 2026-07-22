#!/usr/bin/env bash
# 用途：阶段 04 启动入口；运行 WorldStereo，对轨迹视频做多视角扩展。
# 默认 worldstereo-memory-dmd（与 sh1_stable12k 实验一致）；非 DMD: --model-type worldstereo-memory
set -euo pipefail
STAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$STAGE_DIR/../scripts/env.sh"
source "$HY2_SCRIPT_ROOT/activate_env.sh"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HY2_WORLDSTEREO_MATCH_INPUT_RES="${HY2_WORLDSTEREO_MATCH_INPUT_RES:-1}"
NAME=""; NPROC="${NPROC_PER_NODE:-1}"; MODEL_TYPE="${HY2_WORLDSTEREO_MODEL_TYPE:-worldstereo-memory-dmd}"; FSDP=0; SKIP_EXIST=0
ALIGN_NFRAME="${HY2_ALIGN_NFRAME:-8}"; MAX_REFERENCE="${HY2_MAX_REFERENCE:-8}"
KB_ANOMALY_PERCENTILE="${HY2_KB_ANOMALY_PERCENTILE:-80}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --model-type) MODEL_TYPE="$2"; shift 2 ;;
    --align-nframe) ALIGN_NFRAME="$2"; shift 2 ;;
    --max-reference) MAX_REFERENCE="$2"; shift 2 ;;
    --kb-anomaly-percentile) KB_ANOMALY_PERCENTILE="$2"; shift 2 ;;
    --fsdp) FSDP=1; shift ;;
    --skip-exist) SKIP_EXIST=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$NAME" ]] || { echo "--name required" >&2; exit 2; }
SCENE_DIR="$HY2_RUN_ROOT/$NAME/scene"
STAGE04_VISIBLE_DEVICES="${HY2_STAGE04_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "$STAGE04_VISIBLE_DEVICES" ]]; then
  STAGE04_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC - 1)))
fi
IFS=',' read -r -a STAGE04_DEVICE_LIST <<< "$STAGE04_VISIBLE_DEVICES"
if [[ "${#STAGE04_DEVICE_LIST[@]}" -ne "$NPROC" ]]; then
  echo "Stage04 requires exactly NPROC visible GPUs: nproc=$NPROC CUDA_VISIBLE_DEVICES=$STAGE04_VISIBLE_DEVICES" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$STAGE04_VISIBLE_DEVICES"
python "$HY2_PY_ROOT/require_cuda.py" --stage "stage04 WorldStereo"
cd "$HY2_WORLDGEN_ROOT"
args=(--target_path "$SCENE_DIR" --model_type "$MODEL_TYPE" --align_nframe "$ALIGN_NFRAME" --max_reference "$MAX_REFERENCE" --kb_anomaly_percentile "$KB_ANOMALY_PERCENTILE" --local_files_only)
[[ "$FSDP" == 1 ]] && args+=(--fsdp)
[[ "$SKIP_EXIST" == 1 ]] && args+=(--skip_exist)
{
  echo "[run_04_worldstereo] model_type=$MODEL_TYPE align_nframe=$ALIGN_NFRAME max_reference=$MAX_REFERENCE kb_anomaly_percentile=$KB_ANOMALY_PERCENTILE nproc=$NPROC CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES match_input_res=$HY2_WORLDSTEREO_MATCH_INPUT_RES"
  torchrun --standalone --nproc_per_node "$NPROC" video_gen.py "${args[@]}"
  python "$HY2_PY_ROOT/check_frame_camera_counts.py" --scene-dir "$SCENE_DIR" --result-name "$MODEL_TYPE"
# 使用追加模式保留中断前的进度，便于 --skip-exist 恢复时排查和追踪。
} 2>&1 | tee -a "$HY2_RUN_ROOT/$NAME/logs/04_worldstereo.log"
