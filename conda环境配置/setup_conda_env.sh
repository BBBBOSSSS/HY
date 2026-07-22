#!/usr/bin/env bash
# 用途：复现项目时一键创建 HY2 管线 conda 环境、vLLM 环境，并安装运行依赖。
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "$SETUP_DIR/.." && pwd)"
source "$PIPELINE_ROOT/scripts/env.sh"

HY2_ENV_PATH="${HY2_ENV_PATH:-$HY2_BUNDLE_ROOT/conda-envs/hyworld2}"
VLLM_ENV_PATH="${VLLM_ENV_PATH:-$HY2_BUNDLE_ROOT/conda-envs/vllm_qwen}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1}"
VLLM_VERSION="${VLLM_VERSION:-0.23.0}"
MOGE_COMMIT="${MOGE_COMMIT:-0286b495230a074aadf1c76cc5c679e943e5d1c6}"
FORCE=0
DRY_RUN=0
SKIP_VLLM=0
SKIP_GSPLAT=0
SKIP_PREFLIGHT=0

usage() {
  cat <<'EOF'
用法：
  bash setup_conda_env.sh [选项]

常用选项：
  --hy2-env PATH        HY2 主环境路径，默认 /root/autodl-tmp/conda-envs/hyworld2
  --vllm-env PATH       vLLM 环境路径，默认 /root/autodl-tmp/conda-envs/vllm_qwen
  --python VERSION      Python 版本，默认 3.11
  --torch-index URL     PyTorch wheel 源，默认 https://download.pytorch.org/whl/cu128
  --torch-packages TXT  PyTorch 包版本，默认 torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
  --vllm-version VER    vLLM 版本，默认 0.23.0
  --skip-vllm           不创建 vLLM 环境
  --skip-gsplat         不重编译 gsplat CUDA 扩展
  --skip-preflight      安装后不运行管线预检
  --force               如果环境已存在，先删除再重建
  --dry-run             只打印计划，不实际安装
  -h, --help            显示帮助

示例：
  bash setup_conda_env.sh
  bash setup_conda_env.sh --skip-vllm
  bash setup_conda_env.sh --force
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash setup_conda_env.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hy2-env) HY2_ENV_PATH="$2"; shift 2 ;;
    --vllm-env) VLLM_ENV_PATH="$2"; shift 2 ;;
    --python) PYTHON_VERSION="$2"; shift 2 ;;
    --torch-index) TORCH_INDEX_URL="$2"; shift 2 ;;
    --torch-packages) TORCH_PACKAGES="$2"; shift 2 ;;
    --vllm-version) VLLM_VERSION="$2"; shift 2 ;;
    --skip-vllm) SKIP_VLLM=1; shift ;;
    --skip-gsplat) SKIP_GSPLAT=1; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "找不到 conda 初始化脚本：$CONDA_SH" >&2
  echo "请先安装 Miniconda，或设置 CONDA_SH=/path/to/conda.sh" >&2
  exit 2
fi

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" != 1 ]]; then
    "$@"
  fi
}

run_bash() {
  echo "+ $*"
  if [[ "$DRY_RUN" != 1 ]]; then
    bash -lc "$*"
  fi
}

create_env() {
  local env_path="$1"
  local label="$2"
  if [[ -d "$env_path" && "$FORCE" != 1 ]]; then
    echo "[保留] $label 已存在：$env_path"
    return
  fi
  if [[ -d "$env_path" && "$FORCE" == 1 ]]; then
    run_bash "source '$CONDA_SH' && conda env remove -p '$env_path' -y"
  fi
  mkdir -p "$(dirname "$env_path")"
  run_bash "source '$CONDA_SH' && conda create -p '$env_path' -y python='$PYTHON_VERSION' pip"
}

pip_install() {
  local env_path="$1"
  shift
  run_cmd "$env_path/bin/python" -m pip install "$@"
}

pip_install_optional() {
  local env_path="$1"
  shift
  echo "+ $env_path/bin/python -m pip install $*"
  if [[ "$DRY_RUN" == 1 ]]; then
    return
  fi
  if ! "$env_path/bin/python" -m pip install "$@"; then
    echo "[警告] 可选依赖安装失败：$*"
  fi
}

detect_torch_arch() {
  "$HY2_ENV_PATH/bin/python" - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        print(f"{major}.{minor}")
except Exception:
    pass
PY
}

install_hy2_env() {
  create_env "$HY2_ENV_PATH" "HY2 主环境"
  pip_install "$HY2_ENV_PATH" -U pip setuptools wheel packaging ninja cmake
  pip_install "$HY2_ENV_PATH" --index-url "$TORCH_INDEX_URL" $TORCH_PACKAGES
  pip_install "$HY2_ENV_PATH" \
    diffusers==0.36.0 \
    transformers==5.2.0 \
    accelerate==1.14.0 \
    peft==0.18.1 \
    safetensors==0.8.0 \
    modelscope==1.37.1 \
    "huggingface_hub[cli]==1.20.1" \
    numpy==1.26.4 \
    scipy==1.14.1 \
    omegaconf \
    einops \
    kornia \
    openai \
    easydict \
    timm==1.0.11 \
    Pillow \
    "imageio[ffmpeg]" \
    decord \
    imagesize \
    opencv-python==4.10.0.84 \
    matplotlib==3.10.3 \
    scikit-image==0.25.2 \
    ftfy \
    regex \
    trimesh \
    plyfile==1.0.3 \
    open3d==0.18.0 \
    pycolmap==3.10.0 \
    torchmetrics \
    tensorboard \
    loguru==0.7.3 \
    tqdm \
    viser==1.0.30 \
    tyro==1.0.8 \
    splines \
    pymeshlab==2023.12.post2 \
    rtree==1.4.1 \
    scikit-build-core \
    nanobind \
    pybind11 \
    zim_anything

  # Keep binary CUDA extensions aligned with this project's CUDA 12.8
  # PyTorch runtime.  zim_anything leaves ONNX Runtime unpinned; newer wheels
  # require CUDA 13 and prevent Diffusers from importing.
  pip_install_optional "$HY2_ENV_PATH" \
    numpy==1.26.4 \
    cupy-cuda12x==13.6.0 \
    onnxruntime-gpu==1.20.1

  if [[ -d "$HY2_BUNDLE_ROOT/upstream/moge_src/MoGe-main" ]]; then
    run_bash "git -C '$HY2_BUNDLE_ROOT/upstream/moge_src/MoGe-main' checkout --detach '$MOGE_COMMIT'"
    pip_install "$HY2_ENV_PATH" -e "$HY2_BUNDLE_ROOT/upstream/moge_src/MoGe-main"
  elif [[ -d /root/autodl-tmp/upstream/moge_src/MoGe-main ]]; then
    run_bash "git -C /root/autodl-tmp/upstream/moge_src/MoGe-main checkout --detach '$MOGE_COMMIT'"
    pip_install "$HY2_ENV_PATH" -e /root/autodl-tmp/upstream/moge_src/MoGe-main
  else
    pip_install "$HY2_ENV_PATH" "git+https://github.com/microsoft/MoGe.git@$MOGE_COMMIT"
  fi
}

install_worldgen_extensions() {
  local arch
  local flash_arch
  arch="${TORCH_CUDA_ARCH_LIST:-$(detect_torch_arch || true)}"
  arch="${arch:-9.0}"
  flash_arch="${HY2_FLASH_ATTN_CUDA_ARCHS:-${arch//./}}"

  # The upstream shared installation guide requires one FlashAttention
  # backend.  Use its simpler FlashAttention-2 command and compile against the
  # same CUDA 12.8 toolkit as torch 2.7.1 (important on Blackwell hosts where a
  # newer CUDA toolkit may also be installed).
  run_bash "CUDA_HOME='/usr/local/cuda-12.8' CUDACXX='/usr/local/cuda-12.8/bin/nvcc' FLASH_ATTENTION_FORCE_BUILD='${HY2_FLASH_ATTN_FORCE_BUILD:-TRUE}' FLASH_ATTN_CUDA_ARCHS='$flash_arch' TORCH_CUDA_ARCH_LIST='$arch' MAX_JOBS='${MAX_JOBS:-8}' '$HY2_ENV_PATH/bin/python' -m pip install flash-attn --no-build-isolation"

  # requirements_git.txt: retain the exact upstream revisions.  Prefer the
  # already-downloaded PyTorch3D tree because a full GitHub clone is large and
  # fragile on remote machines.
  pip_install "$HY2_ENV_PATH" --no-build-isolation \
    "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787" \
    "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5" \
    "git+https://github.com/nianticlabs/spz.git@v3.0.0"

  if [[ -d "$HY2_BUNDLE_ROOT/upstream/pytorch3d_src_git/pytorch3d" ]]; then
    run_bash "cd '$HY2_BUNDLE_ROOT/upstream/pytorch3d_src_git' && CUB_HOME='/usr/local/cuda-12.8/include' TORCH_CUDA_ARCH_LIST='$arch' FORCE_CUDA=1 '$HY2_ENV_PATH/bin/python' -m pip install -e . --no-build-isolation"
  else
    run_bash "CUB_HOME='/usr/local/cuda-12.8/include' TORCH_CUDA_ARCH_LIST='$arch' FORCE_CUDA=1 '$HY2_ENV_PATH/bin/python' -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git'"
  fi

  local navmesh_root="$HY2_WORLDGEN_ROOT/third_party/navmesh"
  local recast_root="$HY2_WORLDGEN_ROOT/third_party/recastnavigation"
  if [[ ! -f "$navmesh_root/setup.py" || ! -d "$recast_root/Recast" ]]; then
    echo "[错误] Recast 子模块不完整，请先在源码根目录执行 git submodule update --init --recursive" >&2
    return 1
  fi
  run_bash "cd '$navmesh_root' && RECAST_PATH='$recast_root' '$HY2_ENV_PATH/bin/python' -m pip install . --no-build-isolation"
}

install_vllm_env() {
  if [[ "$SKIP_VLLM" == 1 ]]; then
    echo "[跳过] vLLM 环境"
    return
  fi
  create_env "$VLLM_ENV_PATH" "vLLM 环境"
  pip_install "$VLLM_ENV_PATH" -U pip setuptools wheel packaging
  pip_install "$VLLM_ENV_PATH" --index-url "$TORCH_INDEX_URL" $TORCH_PACKAGES
  pip_install "$VLLM_ENV_PATH" "vllm==$VLLM_VERSION" "transformers==5.12.1" "huggingface_hub[cli]==1.20.1"
}

rebuild_gsplat() {
  if [[ "$SKIP_GSPLAT" == 1 ]]; then
    echo "[跳过] gsplat 重编译"
    return
  fi
  local gsplat_root="$HY2_WORLDGEN_ROOT/third_party/gsplat_maskgaussian"
  if [[ ! -d "$gsplat_root" ]]; then
    echo "[警告] 找不到 gsplat 目录：$gsplat_root"
    return
  fi
  local arch
  arch="${TORCH_CUDA_ARCH_LIST:-$(detect_torch_arch || true)}"
  arch="${arch:-9.0}"
  echo "gsplat 编译架构：$arch"
  run_bash "cd '$gsplat_root' && TORCH_CUDA_ARCH_LIST='$arch' '$HY2_ENV_PATH/bin/python' -m pip install -e . --no-build-isolation"
}

run_preflight() {
  if [[ "$SKIP_PREFLIGHT" == 1 ]]; then
    echo "[跳过] preflight"
    return
  fi
  run_bash "source '$PIPELINE_ROOT/env.sh' && export HY2_VENV='$HY2_ENV_PATH' && source '$PIPELINE_ROOT/activate_env.sh' && python '$PIPELINE_ROOT/python/preflight.py'"
}

echo "== HY2 conda 环境配置 =="
echo "管线目录：$PIPELINE_ROOT"
echo "HY2 主环境：$HY2_ENV_PATH"
echo "vLLM 环境：$VLLM_ENV_PATH"
echo "PyTorch 源：$TORCH_INDEX_URL"
echo "PyTorch 包：$TORCH_PACKAGES"

install_hy2_env
install_worldgen_extensions
install_vllm_env
rebuild_gsplat
run_preflight

echo
echo "环境配置流程结束。"
echo "激活方式："
echo "  source $PIPELINE_ROOT/env.sh"
echo "  source $PIPELINE_ROOT/activate_env.sh"
