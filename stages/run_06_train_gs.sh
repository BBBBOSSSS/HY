#!/usr/bin/env bash
# 用途：阶段 06 启动入口；训练并导出 3DGS，同时后处理最终 PLY。
set -euo pipefail
STAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$STAGE_DIR/../scripts/env.sh"
source "$HY2_SCRIPT_ROOT/activate_env.sh"
if [[ -n "${HY2_CUDA13_ROOT:-}" && -x "$HY2_CUDA13_ROOT/bin/nvcc" ]]; then
  export CUDA_HOME="$HY2_CUDA13_ROOT"
  export CUDA_PATH="$HY2_CUDA13_ROOT"
  export CUDACXX="$HY2_CUDA13_ROOT/bin/nvcc"
  export PATH="$HY2_CUDA13_ROOT/bin:$PATH"
else
  export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
  export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
  export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
fi
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/root/autodl-tmp/.cache/torch_extensions}"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
NAME=""; STEPS=12000; DISABLE_VIEWER=1; EXPORT_MESH="${HY2_EXPORT_MESH:-0}"; NPROC="${NPROC_PER_NODE:-1}"; RESUME_CKPT=""
CLEAN_PLY="${HY2_CLEAN_PLY:-1}"
CLEAN_PLY_OPACITY="${HY2_CLEAN_PLY_OPACITY:-0.93}"
ALIGN_VIEWER_PLY="${HY2_ALIGN_VIEWER_PLY:-1}"
ALIGN_VIEWER_TARGET="${HY2_ALIGN_VIEWER_TARGET:-y-up}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --resume-ckpt) RESUME_CKPT="$2"; shift 2 ;;
    --viewer) DISABLE_VIEWER=0; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$NAME" ]] || { echo "--name required" >&2; exit 2; }
RUN_DIR="$HY2_RUN_ROOT/$NAME"
SCENE_DIR="$RUN_DIR/scene"
RESULT_DIR="$RUN_DIR/gs_result"
# 断点恢复时可在运行目录放置 .hy2_train_steps，以覆盖启动命令中的步数；
# 仅影响当前运行，适合在耗时的前置阶段完成后调整 3DGS 训练预算。
STEPS_OVERRIDE_FILE="$RUN_DIR/.hy2_train_steps"
if [[ -f "$STEPS_OVERRIDE_FILE" ]]; then
  STEPS_OVERRIDE=$(tr -d '[:space:]' < "$STEPS_OVERRIDE_FILE")
  if [[ "$STEPS_OVERRIDE" =~ ^[1-9][0-9]*$ ]]; then
    echo "[run_06_train_gs] applying per-run step override: $STEPS -> $STEPS_OVERRIDE"
    STEPS="$STEPS_OVERRIDE"
  else
    echo "[run_06_train_gs] invalid step override in $STEPS_OVERRIDE_FILE: $STEPS_OVERRIDE" >&2
    exit 2
  fi
fi
MID_STEP_1=$((STEPS/2))
MID_STEP_2=$((STEPS*3/4))
python "$HY2_PY_ROOT/require_cuda.py" --stage "stage06 3DGS training"
cd "$HY2_WORLDGEN_ROOT"
args=(default --data_dir "$SCENE_DIR/gs_data" --result_dir "$RESULT_DIR" --max_steps "$STEPS" --save_steps "$MID_STEP_1" "$MID_STEP_2" "$STEPS" --eval_steps "$STEPS" --ply_steps "$MID_STEP_1" "$MID_STEP_2" "$STEPS" --save_ply --convert_to_spz --disable_video --use_scale_regularization --antialiased --depth_loss --normal_loss --sky_depth_from_pcd --strategy.refine-start-iter 150 --strategy.refine-stop-iter $((STEPS/2)) --strategy.refine-every 100 --strategy.refine-scale2d-stop-iter $((STEPS/2)) --strategy.reset-every 99990 --strategy.grow-grad2d 0.0001 --strategy.prune-scale3d 0.1)
[[ "$DISABLE_VIEWER" == 1 ]] && args+=(--disable_viewer)
[[ "$EXPORT_MESH" == 1 ]] && args+=(--export_mesh)
[[ -n "$RESUME_CKPT" ]] && args+=(--resume-ckpt "$RESUME_CKPT")
if [[ "$NPROC" -gt 1 ]]; then
  echo "[run_06_train_gs] NPROC=$NPROC: world_gs_trainer manages multi-GPU workers internally; launching without outer torchrun."
fi
python -m world_gs_trainer "${args[@]}" 2>&1 | tee "$RUN_DIR/logs/06_train_gs.log"

if [[ "$CLEAN_PLY" == 1 ]]; then
  PLY_DIR="$RESULT_DIR/ply"
  FINAL_PLY=$(find "$PLY_DIR" -maxdepth 1 -type f -name 'point_cloud_*.ply' ! -name '*_clean_opacity_*.ply' 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -n "$FINAL_PLY" ]]; then
    {
      echo "[run_06_train_gs] cleaning final PLY by opacity >= $CLEAN_PLY_OPACITY"
      python "$HY2_PY_ROOT/clean_gaussian_ply_opacity.py" --input "$FINAL_PLY" --threshold "$CLEAN_PLY_OPACITY"
    } 2>&1 | tee -a "$RUN_DIR/logs/06_train_gs.log"
  else
    echo "[run_06_train_gs] no final PLY found under $PLY_DIR; skip opacity cleaning" | tee -a "$RUN_DIR/logs/06_train_gs.log"
  fi
fi

if [[ "$ALIGN_VIEWER_PLY" == 1 ]]; then
  PLY_DIR="$RESULT_DIR/ply"
  FINAL_PLY=$(find "$PLY_DIR" -maxdepth 1 -type f -name 'point_cloud_*.ply' ! -name '*_clean_opacity_*.ply' ! -name '*_viewer_*.ply' 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -n "$FINAL_PLY" ]]; then
    {
      echo "[run_06_train_gs] aligning final PLY for browser viewer target=$ALIGN_VIEWER_TARGET"
      python "$HY2_PY_ROOT/align_gaussian_ply_for_viewer.py" --input "$FINAL_PLY" --target "$ALIGN_VIEWER_TARGET"
      CLEAN_SUFFIX=$(printf "%s" "$CLEAN_PLY_OPACITY" | sed 's/[^0-9]/_/g')
      CLEANED_PLY="${FINAL_PLY%.ply}_clean_opacity_${CLEAN_SUFFIX}.ply"
      if [[ -f "$CLEANED_PLY" ]]; then
        echo "[run_06_train_gs] aligning cleaned final PLY for browser viewer target=$ALIGN_VIEWER_TARGET"
        python "$HY2_PY_ROOT/align_gaussian_ply_for_viewer.py" --input "$CLEANED_PLY" --target "$ALIGN_VIEWER_TARGET"
      fi
    } 2>&1 | tee -a "$RUN_DIR/logs/06_train_gs.log"
  else
    echo "[run_06_train_gs] no final PLY found under $PLY_DIR; skip viewer alignment" | tee -a "$RUN_DIR/logs/06_train_gs.log"
  fi
fi
