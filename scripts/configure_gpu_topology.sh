#!/usr/bin/env bash
# Source after scripts/env.sh.  Exports a deterministic stage-to-GPU mapping.
# HY2_PIPELINE_GPU_MODE: auto (default), single, or dual.

_hy2_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk 'NF {n++} END {print n+0}')"
HY2_PIPELINE_GPU_MODE="${HY2_PIPELINE_GPU_MODE:-auto}"
if [[ "$HY2_PIPELINE_GPU_MODE" == "auto" ]]; then
  if [[ "$_hy2_gpu_count" -ge 2 ]]; then
    HY2_PIPELINE_GPU_MODE="dual"
  else
    HY2_PIPELINE_GPU_MODE="single"
  fi
fi

case "$HY2_PIPELINE_GPU_MODE" in
  single)
    # One GPU: stage-local models and vLLM share this device during the VLM
    # calls in Stages 02/03.  Cap vLLM's KV cache so SAM/WorldNav still has
    # ample headroom; release vLLM before WorldStereo/GS training.
    export HY2_VLLM_CUDA_VISIBLE_DEVICES=0
    export HY2_STAGE02_CUDA_VISIBLE_DEVICES=0
    export HY2_STAGE03_CUDA_VISIBLE_DEVICES=0
    export HY2_STAGE04_CUDA_VISIBLE_DEVICES=0
    export HY2_STAGE05_CUDA_VISIBLE_DEVICES=0
    export HY2_STAGE06_CUDA_VISIBLE_DEVICES=0
    export HY2_STAGE04_NPROC=1
    export HY2_STAGE05_NPROC=1
    export HY2_STAGE06_NPROC=1
    export HY2_RENDER_NPROC=1
    export HY2_VLLM_GPU_UTIL="${HY2_VLLM_GPU_UTIL_SINGLE:-0.30}"
    ;;
  dual)
    # Two GPUs: vLLM lives on GPU 1 while VLM-dependent navigation/rendering
    # stay on GPU 0. Release vLLM before the two-GPU GS phases.
    export HY2_VLLM_CUDA_VISIBLE_DEVICES=1
    export HY2_STAGE02_CUDA_VISIBLE_DEVICES=0
    export HY2_STAGE03_CUDA_VISIBLE_DEVICES=0
    export HY2_STAGE04_CUDA_VISIBLE_DEVICES=0
    export HY2_STAGE05_CUDA_VISIBLE_DEVICES=0,1
    export HY2_STAGE06_CUDA_VISIBLE_DEVICES=0,1
    export HY2_STAGE04_NPROC=1
    export HY2_STAGE05_NPROC=2
    export HY2_STAGE06_NPROC=2
    export HY2_RENDER_NPROC=1
    ;;
  *)
    echo "Invalid HY2_PIPELINE_GPU_MODE=$HY2_PIPELINE_GPU_MODE (use auto|single|dual)" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

export HY2_PIPELINE_GPU_MODE
echo "[gpu_topology] mode=$HY2_PIPELINE_GPU_MODE detected_gpus=$_hy2_gpu_count vllm=$HY2_VLLM_CUDA_VISIBLE_DEVICES stage02=$HY2_STAGE02_CUDA_VISIBLE_DEVICES stage03=$HY2_STAGE03_CUDA_VISIBLE_DEVICES stage04=$HY2_STAGE04_CUDA_VISIBLE_DEVICES stage05=$HY2_STAGE05_CUDA_VISIBLE_DEVICES/$HY2_STAGE05_NPROC stage06=$HY2_STAGE06_CUDA_VISIBLE_DEVICES/$HY2_STAGE06_NPROC"
