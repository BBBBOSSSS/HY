#!/usr/bin/env bash
# 用途：共享环境配置；定义管线路径、模型路径和默认运行参数，阶段脚本运行前会加载它。
set -euo pipefail

_HY2_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HY2_PIPELINE_ROOT="${HY2_PIPELINE_ROOT:-$(cd "$_HY2_ENV_DIR/.." && pwd)}"
export HY2_PIPELINES_ROOT="${HY2_PIPELINES_ROOT:-$(cd "$HY2_PIPELINE_ROOT/.." && pwd)}"
export HY2_BUNDLE_ROOT="${HY2_BUNDLE_ROOT:-$HY2_PIPELINES_ROOT/_bundle_deps}"
export HY2_STAGE_ROOT="${HY2_STAGE_ROOT:-$HY2_PIPELINE_ROOT/stages}"
export HY2_SCRIPT_ROOT="${HY2_SCRIPT_ROOT:-$HY2_PIPELINE_ROOT/scripts}"
export HY2_PY_ROOT="${HY2_PY_ROOT:-$HY2_PIPELINE_ROOT/python}"

if [[ -d "${HY2_CODE_ROOT:-$HY2_BUNDLE_ROOT/upstream/HY-World-2.0-gh}/hyworld2/worldgen" ]]; then
  export HY2_CODE_ROOT="${HY2_CODE_ROOT:-$HY2_BUNDLE_ROOT/upstream/HY-World-2.0-gh}"
elif [[ -d "${HY2_CODE_ROOT:-/root/autodl-tmp/upstream/HY-World-2.0}/hyworld2/worldgen" ]]; then
  export HY2_CODE_ROOT="${HY2_CODE_ROOT:-/root/autodl-tmp/upstream/HY-World-2.0}"
else
  export HY2_CODE_ROOT="${HY2_CODE_ROOT:-/root/autodl-tmp/upstream/HY-World-2.0-gh}"
fi

export HY2_WORLDGEN_ROOT="$HY2_CODE_ROOT/hyworld2/worldgen"
export HY2_RUN_ROOT="${HY2_RUN_ROOT:-/root/autodl-tmp/outputs/hy2_worldgen_runs}"
if [[ -d "${HY2_VENV:-$HY2_BUNDLE_ROOT/conda-envs/hyworld2}" ]]; then
  export HY2_VENV="${HY2_VENV:-$HY2_BUNDLE_ROOT/conda-envs/hyworld2}"
else
  export HY2_VENV="${HY2_VENV:-/root/autodl-tmp/conda-envs/hyworld2}"
fi
export HY2_CUDA13_ROOT="${HY2_CUDA13_ROOT:-$HY2_VENV/lib/python3.11/site-packages/nvidia/cu13}"
if [[ -d "$HY2_CUDA13_ROOT" ]]; then
  export CUDA_HOME="${CUDA_HOME:-$HY2_CUDA13_ROOT}"
  export CUDA_PATH="${CUDA_PATH:-$HY2_CUDA13_ROOT}"
  export PATH="$HY2_CUDA13_ROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$HY2_CUDA13_ROOT/lib:$HY2_VENV/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="$HY2_VENV/lib:$HY2_VENV/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ -d "${MODEL_ROOT:-$HY2_BUNDLE_ROOT/models}" ]]; then
  export MODEL_ROOT="${MODEL_ROOT:-$HY2_BUNDLE_ROOT/models}"
else
  export MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/models}"
fi
export HY_PANO_BACKEND="${HY_PANO_BACKEND:-qwen-lora}"
export HY_PANO_MODEL_PATH="${HY_PANO_MODEL_PATH:-$MODEL_ROOT/HY-World-2.0-modelscope-full/HY-Pano-2.0}"
export QWEN_PANO_BASE="${QWEN_PANO_BASE:-$MODEL_ROOT/Qwen-Image-Edit-2509}"
export HY_PANO_LORA_REPO="${HY_PANO_LORA_REPO:-$MODEL_ROOT/HY-World-2.0-modelscope-full}"
export HY_PANO_LORA_SUBFOLDER="${HY_PANO_LORA_SUBFOLDER:-HY-Pano-2.0}"
export HY_PANO_DEVICE_MAP="${HY_PANO_DEVICE_MAP:-auto}"
export HY_PANO_MAX_GPU_MEMORY="${HY_PANO_MAX_GPU_MEMORY:-}"
export HY_PANO_MAX_CPU_MEMORY="${HY_PANO_MAX_CPU_MEMORY:-}"
export HY_PANO_OFFLOAD_DIR="${HY_PANO_OFFLOAD_DIR:-$HY2_RUN_ROOT/_hf_offload/hy_pano2}"
export HY_PANO_TRUE_CFG_SCALE="${HY_PANO_TRUE_CFG_SCALE:-7.5}"

export WORLDSTEREO_REPO_ID="${WORLDSTEREO_REPO_ID:-$MODEL_ROOT/worldstereo}"
export WORLDSTEREO_BASE_MODEL_PATH="${WORLDSTEREO_BASE_MODEL_PATH:-$MODEL_ROOT/Wan2.1-I2V-14B-720P-Diffusers}"
export WAN_BASE_MODEL_PATH="$WORLDSTEREO_BASE_MODEL_PATH"
export SAM3_REPO_ID="${SAM3_REPO_ID:-$MODEL_ROOT/sam3}"
export MOGE_MODEL_ID="${MOGE_MODEL_ID:-$MODEL_ROOT/moge-2-vitl-normal-official/model.pt}"
export UNI3C_CONTROLNET_PATH="${UNI3C_CONTROLNET_PATH:-$MODEL_ROOT/Uni3C/controlnet.pth}"
export WORLDMIRROR_MODEL_PATH="${WORLDMIRROR_MODEL_PATH:-$MODEL_ROOT/hy-worldmirror}"

export QWEN3_VL_MODEL_PATH="${QWEN3_VL_MODEL_PATH:-$MODEL_ROOT/Qwen3-VL-8B-Instruct}"
export DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$MODEL_ROOT/dinov2-base}"
export NAVER_IV_ZIM_ANYTHING_VITL_PATH="${NAVER_IV_ZIM_ANYTHING_VITL_PATH:-$MODEL_ROOT/zim-anything-vitl}"
export IDEA_RESEARCH_GROUNDING_DINO_TINY_PATH="${IDEA_RESEARCH_GROUNDING_DINO_TINY_PATH:-$MODEL_ROOT/grounding-dino-tiny}"
export HY2_LLM_ADDR="${HY2_LLM_ADDR:-127.0.0.1}"
export HY2_LLM_PORT="${HY2_LLM_PORT:-8000}"
export HY2_LLM_NAME="${HY2_LLM_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,0.0.0.0}"
export no_proxy="${no_proxy:-127.0.0.1,localhost,0.0.0.0}"
# WorldStereo：默认 DMD（与 sh1_stable12k 实验一致；memory 在本场景无显著提升）
# 若要用非 DMD: export HY2_WORLDSTEREO_MODEL_TYPE=worldstereo-memory
export HY2_WORLDSTEREO_MODEL_TYPE="${HY2_WORLDSTEREO_MODEL_TYPE:-worldstereo-memory-dmd}"
# Stage06：MaskGaussian / anchor（1=开）；opacity 过滤导出默认关
export HY2_USE_MASK_GAUSSIAN="${HY2_USE_MASK_GAUSSIAN:-1}"
export HY2_USE_ANCHOR_PROTECTION="${HY2_USE_ANCHOR_PROTECTION:-1}"
export HY2_CLEAN_PLY="${HY2_CLEAN_PLY:-0}"
# Stage06 默认 = sh1_stable12k 配置（当前最优验证结果）
# sh1 + LPIPS 0.2/0.1 + 弱 densify + 12k step + 不强制 2× 导出
export HY2_SH_DEGREE="${HY2_SH_DEGREE:-1}"
export HY2_SH_DEGREE_INTERVAL="${HY2_SH_DEGREE_INTERVAL:-1500}"
export HY2_LPIPS_LAMBDA1="${HY2_LPIPS_LAMBDA1:-0.2}"
export HY2_LPIPS_LAMBDA2="${HY2_LPIPS_LAMBDA2:-0.1}"
export HY2_GROW_GRAD2D="${HY2_GROW_GRAD2D:-0.00025}"
export HY2_REFINE_EVERY="${HY2_REFINE_EVERY:-150}"
export HY2_REFINE_STOP_FRAC="${HY2_REFINE_STOP_FRAC:-0.40}"
export HY2_GS_STEPS="${HY2_GS_STEPS:-12000}"
export HY2_EXPORT_RENDER="${HY2_EXPORT_RENDER:-0}"
export HY2_EXPORT_SCALE="${HY2_EXPORT_SCALE:-1}"

export HY2_VLLM_TP_SIZE="${HY2_VLLM_TP_SIZE:-1}"
export HY2_VLLM_CUDA_VISIBLE_DEVICES="${HY2_VLLM_CUDA_VISIBLE_DEVICES:-}"
export HY2_STAGE03_CUDA_VISIBLE_DEVICES="${HY2_STAGE03_CUDA_VISIBLE_DEVICES:-}"
export HY2_STAGE04_CUDA_VISIBLE_DEVICES="${HY2_STAGE04_CUDA_VISIBLE_DEVICES:-}"
export HY2_VLLM_GPU_UTIL="${HY2_VLLM_GPU_UTIL:-0.90}"
export HY2_VLLM_MAX_MODEL_LEN="${HY2_VLLM_MAX_MODEL_LEN:-8192}"
export HY2_VLLM_MM_LIMIT_IMAGE="${HY2_VLLM_MM_LIMIT_IMAGE:-8}"
export HY2_VLLM_EXTRA_ARGS="${HY2_VLLM_EXTRA_ARGS:---enforce-eager --no-async-scheduling --no-enable-chunked-prefill --max-num-seqs 1 --generation-config vllm}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

HY2_GSPLAT_ROOT="${HY2_GSPLAT_ROOT:-$HY2_WORLDGEN_ROOT/third_party/gsplat_maskgaussian}"
export HY2_GSPLAT_ROOT
export PYTHONPATH="$HY2_CODE_ROOT:$HY2_CODE_ROOT/hyworld2:$HY2_WORLDGEN_ROOT:$HY2_GSPLAT_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HY2_OFFLINE_AUX="${HY2_OFFLINE_AUX:-1}"
export WORLDGEN_SKIP_VLLM="${WORLDGEN_SKIP_VLLM:-0}"
export WORLDGEN_FALLBACK_CAPTION="${WORLDGEN_FALLBACK_CAPTION:-A coherent 360-degree scene with stable geometry, continuous navigable ground or floor, fixed landmark identity, consistent lighting, realistic scale, sharp static details, and coherent depth across all frames.}"
export OMP_NUM_THREADS=1

mkdir -p "$HY2_RUN_ROOT"
