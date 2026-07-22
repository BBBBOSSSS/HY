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
# 默认步数与 sh1_stable12k 一致（可用 HY2_GS_STEPS / --steps 覆盖）
NAME=""; STEPS="${HY2_GS_STEPS:-12000}"; DISABLE_VIEWER=1; EXPORT_MESH="${HY2_EXPORT_MESH:-0}"; NPROC="${NPROC_PER_NODE:-1}"; RESUME_CKPT=""
# 默认不做 opacity 过滤导出（避免镂空/变糊）；需要时: HY2_CLEAN_PLY=1
CLEAN_PLY="${HY2_CLEAN_PLY:-0}"
CLEAN_PLY_OPACITY="${HY2_CLEAN_PLY_OPACITY:-0.93}"
ALIGN_VIEWER_PLY="${HY2_ALIGN_VIEWER_PLY:-1}"
ALIGN_VIEWER_TARGET="${HY2_ALIGN_VIEWER_TARGET:-y-up}"
# 对齐官方 worldgen README：MaskGaussian + anchor protection
USE_MASK_GAUSSIAN="${HY2_USE_MASK_GAUSSIAN:-1}"
USE_ANCHOR_PROTECTION="${HY2_USE_ANCHOR_PROTECTION:-1}"
# GS 超参（可用环境变量覆盖；稳妥默认：sh1 + 原版 LPIPS + 较弱 densify）
# 官方 Config: sh_degree / sh_degree_interval / lpips_lambda1 / lpips_lambda2 / strategy.*
SH_DEGREE="${HY2_SH_DEGREE:-1}"
SH_DEGREE_INTERVAL="${HY2_SH_DEGREE_INTERVAL:-1500}"
LPIPS_LAMBDA1="${HY2_LPIPS_LAMBDA1:-0.2}"
LPIPS_LAMBDA2="${HY2_LPIPS_LAMBDA2:-0.1}"
# densify：grow-grad2d 越大越难增生；refine-every 越大越稀疏；stop-frac 越小越早停 densify
GROW_GRAD2D="${HY2_GROW_GRAD2D:-0.00025}"
REFINE_EVERY="${HY2_REFINE_EVERY:-150}"
REFINE_STOP_FRAC="${HY2_REFINE_STOP_FRAC:-0.40}"
PRUNE_SCALE3D="${HY2_PRUNE_SCALE3D:-0.1}"
# 训练后固定分辨率导出（默认关；需要时 HY2_EXPORT_RENDER=1，scale=1 为训练分辨率）
EXPORT_RENDER="${HY2_EXPORT_RENDER:-0}"
EXPORT_SCALE="${HY2_EXPORT_SCALE:-1}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --resume-ckpt) RESUME_CKPT="$2"; shift 2 ;;
    --viewer) DISABLE_VIEWER=0; shift ;;
    --no-mask-gaussian) USE_MASK_GAUSSIAN=0; shift ;;
    --no-anchor-protection) USE_ANCHOR_PROTECTION=0; shift ;;
    --sh-degree) SH_DEGREE="$2"; shift 2 ;;
    --sh-degree-interval) SH_DEGREE_INTERVAL="$2"; shift 2 ;;
    --lpips-lambda1) LPIPS_LAMBDA1="$2"; shift 2 ;;
    --lpips-lambda2) LPIPS_LAMBDA2="$2"; shift 2 ;;
    --export-scale) EXPORT_SCALE="$2"; shift 2 ;;
    --no-export-render) EXPORT_RENDER=0; shift ;;
    --export-render) EXPORT_RENDER=1; shift ;;
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
# densify 停止步数（默认 40% steps，减弱后期猛增点）
REFINE_STOP_ITER=$(python3 - <<PY
steps=int("$STEPS"); frac=float("$REFINE_STOP_FRAC")
print(max(500, int(steps * frac)))
PY
)
python "$HY2_PY_ROOT/require_cuda.py" --stage "stage06 3DGS training"
cd "$HY2_WORLDGEN_ROOT"
args=(default --data_dir "$SCENE_DIR/gs_data" --result_dir "$RESULT_DIR" --max_steps "$STEPS" --save_steps "$MID_STEP_1" "$MID_STEP_2" "$STEPS" --eval_steps "$STEPS" --ply_steps "$MID_STEP_1" "$MID_STEP_2" "$STEPS" --save_ply --convert_to_spz --disable_video --use_scale_regularization --antialiased --depth_loss --normal_loss --sky_depth_from_pcd --strategy.refine-start-iter 150 --strategy.refine-stop-iter "$REFINE_STOP_ITER" --strategy.refine-every "$REFINE_EVERY" --strategy.refine-scale2d-stop-iter "$REFINE_STOP_ITER" --strategy.reset-every 99990 --strategy.grow-grad2d "$GROW_GRAD2D" --strategy.prune-scale3d "$PRUNE_SCALE3D")
args+=(--sh_degree "$SH_DEGREE" --sh_degree_interval "$SH_DEGREE_INTERVAL")
args+=(--lpips_lambda1 "$LPIPS_LAMBDA1" --lpips_lambda2 "$LPIPS_LAMBDA2")
# 与官方 README Stage5 对齐的 MaskGaussian / anchor 开关
if [[ "$USE_MASK_GAUSSIAN" == 1 ]]; then
  args+=(--use_mask_gaussian --mask_export_stochastic)
fi
if [[ "$USE_ANCHOR_PROTECTION" == 1 ]]; then
  # 官方：保护训练期 anchor，导出时不做 anchor 特殊保留（与 README 一致）
  args+=(--use_anchor_protection --no-mask-export-anchor-protection)
fi
[[ "$DISABLE_VIEWER" == 1 ]] && args+=(--disable_viewer)
[[ "$EXPORT_MESH" == 1 ]] && args+=(--export_mesh)
[[ -n "$RESUME_CKPT" ]] && args+=(--resume-ckpt "$RESUME_CKPT")
if [[ "$NPROC" -gt 1 ]]; then
  echo "[run_06_train_gs] NPROC=$NPROC: world_gs_trainer manages multi-GPU workers internally; launching without outer torchrun."
fi
{
  echo "[run_06_train_gs] use_mask_gaussian=$USE_MASK_GAUSSIAN use_anchor_protection=$USE_ANCHOR_PROTECTION clean_ply=$CLEAN_PLY steps=$STEPS"
  echo "[run_06_train_gs] sh_degree=$SH_DEGREE sh_degree_interval=$SH_DEGREE_INTERVAL lpips_lambda1=$LPIPS_LAMBDA1 lpips_lambda2=$LPIPS_LAMBDA2"
  echo "[run_06_train_gs] densify grow_grad2d=$GROW_GRAD2D refine_every=$REFINE_EVERY refine_stop=$REFINE_STOP_ITER export_render=$EXPORT_RENDER scale=$EXPORT_SCALE"
  echo "[run_06_train_gs] args: ${args[*]}"
  python -m world_gs_trainer "${args[@]}"
} 2>&1 | tee "$RUN_DIR/logs/06_train_gs.log"

# 可选：opacity 过滤导出（默认关闭，保留完整未过滤 PLY）
if [[ "$CLEAN_PLY" == 1 ]]; then
  PLY_DIR="$RESULT_DIR/ply"
  FINAL_PLY=$(find "$PLY_DIR" -maxdepth 1 -type f -name 'point_cloud_*.ply' ! -name '*_clean_opacity_*.ply' ! -name '*_viewer_*.ply' 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -n "$FINAL_PLY" ]]; then
    {
      echo "[run_06_train_gs] cleaning final PLY by opacity >= $CLEAN_PLY_OPACITY"
      python "$HY2_PY_ROOT/clean_gaussian_ply_opacity.py" --input "$FINAL_PLY" --threshold "$CLEAN_PLY_OPACITY"
    } 2>&1 | tee -a "$RUN_DIR/logs/06_train_gs.log"
  else
    echo "[run_06_train_gs] no final PLY found under $PLY_DIR; skip opacity cleaning" | tee -a "$RUN_DIR/logs/06_train_gs.log"
  fi
else
  echo "[run_06_train_gs] skip opacity clean export (HY2_CLEAN_PLY=$CLEAN_PLY; export raw PLY only)" | tee -a "$RUN_DIR/logs/06_train_gs.log"
fi

if [[ "$ALIGN_VIEWER_PLY" == 1 ]]; then
  PLY_DIR="$RESULT_DIR/ply"
  # 只对齐原始最终 PLY（不依赖 clean_opacity 产物）
  FINAL_PLY=$(find "$PLY_DIR" -maxdepth 1 -type f -name 'point_cloud_*.ply' ! -name '*_clean_opacity_*.ply' ! -name '*_viewer_*.ply' 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -n "$FINAL_PLY" ]]; then
    {
      echo "[run_06_train_gs] aligning final PLY for browser viewer target=$ALIGN_VIEWER_TARGET"
      python "$HY2_PY_ROOT/align_gaussian_ply_for_viewer.py" --input "$FINAL_PLY" --target "$ALIGN_VIEWER_TARGET"
      if [[ "$CLEAN_PLY" == 1 ]]; then
        CLEAN_SUFFIX=$(printf "%s" "$CLEAN_PLY_OPACITY" | sed 's/[^0-9]/_/g')
        CLEANED_PLY="${FINAL_PLY%.ply}_clean_opacity_${CLEAN_SUFFIX}.ply"
        if [[ -f "$CLEANED_PLY" ]]; then
          echo "[run_06_train_gs] aligning cleaned final PLY for browser viewer target=$ALIGN_VIEWER_TARGET"
          python "$HY2_PY_ROOT/align_gaussian_ply_for_viewer.py" --input "$CLEANED_PLY" --target "$ALIGN_VIEWER_TARGET"
        fi
      fi
    } 2>&1 | tee -a "$RUN_DIR/logs/06_train_gs.log"
  else
    echo "[run_06_train_gs] no final PLY found under $PLY_DIR; skip viewer alignment" | tee -a "$RUN_DIR/logs/06_train_gs.log"
  fi
fi

# 训练后按训练分辨率或 2× 导出固定视角图（避免 viewer 无脑放大）
if [[ "$EXPORT_RENDER" == 1 ]]; then
  CKPT=$(find "$RESULT_DIR/ckpts" -maxdepth 1 -type f -name 'ckpt_*_rank0.pt' 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -n "$CKPT" && -d "$SCENE_DIR/gs_data" ]]; then
    {
      echo "[run_06_train_gs] fixed-res export scale=${EXPORT_SCALE}x sh_degree=$SH_DEGREE ckpt=$CKPT"
      python "$HY2_PY_ROOT/export_gs_fixed_res.py" \
        --data-dir "$SCENE_DIR/gs_data" \
        --result-dir "$RESULT_DIR" \
        --ckpt "$CKPT" \
        --out-dir "$RESULT_DIR/export_fixed_res" \
        --scale "$EXPORT_SCALE" \
        --sh-degree "$SH_DEGREE" \
        --max-views 12
    } 2>&1 | tee -a "$RUN_DIR/logs/06_train_gs.log"
  else
    echo "[run_06_train_gs] skip fixed-res export (missing ckpt or gs_data)" | tee -a "$RUN_DIR/logs/06_train_gs.log"
  fi
fi
