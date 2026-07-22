#!/usr/bin/env bash
# 用途：复现项目时一键从 ModelScope 下载 HY2 管线需要的模型，并放到管线默认读取路径。
set -euo pipefail

DOWNLOAD_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "$DOWNLOAD_DIR/.." && pwd)"
source "$PIPELINE_ROOT/scripts/env.sh"

MODEL_ROOT="${MODEL_ROOT:-$HY2_BUNDLE_ROOT/models}"
MAX_WORKERS="${MAX_WORKERS:-8}"
DRY_RUN=0
SKIP_PREFLIGHT=0
ONLY=""
FULL_HY_PANO=0

# 这些 repo id 都可以在运行前用环境变量覆盖，方便复现者切到自己的 ModelScope 镜像仓库。
HY_WORLD_MS_REPO_ID="${HY_WORLD_MS_REPO_ID:-Tencent-Hunyuan/HY-World-2.0}"
WORLDMIRROR_MS_REPO_ID="${WORLDMIRROR_MS_REPO_ID:-$HY_WORLD_MS_REPO_ID}"
QWEN_IMAGE_EDIT_MS_REPO_ID="${QWEN_IMAGE_EDIT_MS_REPO_ID:-Qwen/Qwen-Image-Edit-2509}"
QWEN3_VL_MS_REPO_ID="${QWEN3_VL_MS_REPO_ID:-Qwen/Qwen3-VL-8B-Instruct}"
WAN_I2V_MS_REPO_ID="${WAN_I2V_MS_REPO_ID:-Wan-AI/Wan2.1-I2V-14B-720P-Diffusers}"
SAM3_MS_REPO_ID="${SAM3_MS_REPO_ID:-facebook/sam3}"
DINOV2_MS_REPO_ID="${DINOV2_MS_REPO_ID:-facebook/dinov2-base}"
GROUNDING_DINO_MS_REPO_ID="${GROUNDING_DINO_MS_REPO_ID:-IDEA-Research/grounding-dino-tiny}"
MOGE_MS_REPO_ID="${MOGE_MS_REPO_ID:-bluestone25/moge-vitl}"

# 这两个没有确认到稳定的公开 ModelScope 同名仓库；默认走 HF 协议，并优先使用国内镜像端点。
WORLDSTEREO_HF_REPO_ID="${WORLDSTEREO_HF_REPO_ID:-hanshanxue/WorldStereo}"
UNI3C_HF_REPO_ID="${UNI3C_HF_REPO_ID:-ewrfcas/Uni3C}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
ZIM_ANYTHING_MS_REPO_ID="${ZIM_ANYTHING_MS_REPO_ID:-}"

usage() {
  cat <<'EOF'
用法：
  bash model_download.sh [选项]

常用选项：
  --model-root PATH     模型根目录，默认 pipelines/_bundle_deps/models
  --only LIST           只下载指定项，逗号分隔。例如：qwen-image,qwen3-vl,hy-pano-lora
  --full-hy-pano        下载 HY-Pano-2.0 完整模型；默认只下载 LoRA 权重
  --max-workers N       ModelScope 并发下载数，默认 8
  --hf-endpoint URL     Hugging Face 下载端点，默认 https://hf-mirror.com
  --dry-run             只打印下载计划，不真正下载
  --skip-preflight      下载后不运行管线预检
  -h, --help            显示帮助

可用名称：
  hy-pano-lora, qwen-image, qwen3-vl, wan-i2v, sam3, dinov2, grounding-dino, moge,
  worldmirror, worldstereo, uni3c, zim-anything

可覆盖的下载源变量：
  HY_WORLD_MS_REPO_ID
  WORLDMIRROR_MS_REPO_ID
  QWEN_IMAGE_EDIT_MS_REPO_ID
  QWEN3_VL_MS_REPO_ID
  WAN_I2V_MS_REPO_ID
  SAM3_MS_REPO_ID
  DINOV2_MS_REPO_ID
  GROUNDING_DINO_MS_REPO_ID
  MOGE_MS_REPO_ID
  WORLDSTEREO_HF_REPO_ID
  UNI3C_HF_REPO_ID
  HF_ENDPOINT
  ZIM_ANYTHING_MS_REPO_ID
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-root) MODEL_ROOT="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --full-hy-pano) FULL_HY_PANO=1; shift ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --hf-endpoint) HF_ENDPOINT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

export MODEL_ROOT
mkdir -p "$MODEL_ROOT"

has_item() {
  local item="$1"
  [[ -z "$ONLY" ]] && return 0
  case ",$ONLY," in
    *",$item,"*) return 0 ;;
    *) return 1 ;;
  esac
}

download_ms() {
  local name="$1"
  local repo_id="$2"
  local local_dir="$3"
  local allow_patterns="${4:-}"

  if ! has_item "$name"; then
    return
  fi
  if [[ -z "$repo_id" ]]; then
    echo "[跳过] $name：没有配置 ModelScope repo id。可设置 ${name//-/_}_MS_REPO_ID 或查看 README。"
    return
  fi

  echo
  echo "== 下载 $name =="
  echo "repo: $repo_id"
  echo "目标: $local_dir"
  [[ -n "$allow_patterns" ]] && echo "文件过滤: $allow_patterns"
  if [[ "$DRY_RUN" == 1 ]]; then
    return
  fi

  REPO_ID="$repo_id" LOCAL_DIR="$local_dir" ALLOW_PATTERNS="$allow_patterns" MAX_WORKERS="$MAX_WORKERS" python - <<'PY'
import os
from modelscope.hub.snapshot_download import snapshot_download

repo_id = os.environ["REPO_ID"]
local_dir = os.environ["LOCAL_DIR"]
allow_patterns = os.environ.get("ALLOW_PATTERNS", "").strip()
max_workers = int(os.environ.get("MAX_WORKERS", "8"))
patterns = [p.strip() for p in allow_patterns.split(",") if p.strip()] or None

snapshot_download(
    repo_id,
    local_dir=local_dir,
    allow_patterns=patterns,
    max_workers=max_workers,
)
print(f"[完成] {repo_id} -> {local_dir}")
PY
}

download_hf() {
  local name="$1"
  local repo_id="$2"
  local local_dir="$3"
  local allow_patterns="${4:-}"

  if ! has_item "$name"; then
    return
  fi

  echo
  echo "== 下载 $name =="
  echo "repo: $repo_id"
  echo "目标: $local_dir"
  echo "HF endpoint: $HF_ENDPOINT"
  [[ -n "$allow_patterns" ]] && echo "文件过滤: $allow_patterns"
  if [[ "$DRY_RUN" == 1 ]]; then
    return
  fi

  REPO_ID="$repo_id" LOCAL_DIR="$local_dir" ALLOW_PATTERNS="$allow_patterns" HF_ENDPOINT="$HF_ENDPOINT" MAX_WORKERS="$MAX_WORKERS" python - <<'PY'
import os
import subprocess
import sys

try:
    from huggingface_hub import snapshot_download
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])
    from huggingface_hub import snapshot_download

repo_id = os.environ["REPO_ID"]
local_dir = os.environ["LOCAL_DIR"]
allow_patterns = os.environ.get("ALLOW_PATTERNS", "").strip()
max_workers = int(os.environ.get("MAX_WORKERS", "8"))
patterns = [p.strip() for p in allow_patterns.split(",") if p.strip()] or None

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    allow_patterns=patterns,
    max_workers=max_workers,
)
print(f"[完成] {repo_id} -> {local_dir}")
PY
}

need_file() {
  local label="$1"
  local path="$2"
  if [[ -e "$path" ]]; then
    echo "[OK] $label: $path"
  else
    echo "[缺失] $label: $path"
    MISSING_ITEMS+=("$label")
  fi
}

echo "模型根目录：$MODEL_ROOT"
echo "管线目录：$PIPELINE_ROOT"

HY_PANO_PATTERNS="HY-Pano-2.0/pytorch_lora_weights.safetensors"
if [[ "$FULL_HY_PANO" == 1 ]]; then
  HY_PANO_PATTERNS="HY-Pano-2.0/*"
fi

download_ms "hy-pano-lora" "$HY_WORLD_MS_REPO_ID" "$MODEL_ROOT/HY-World-2.0-modelscope-full" "$HY_PANO_PATTERNS"
download_ms "worldmirror" "$WORLDMIRROR_MS_REPO_ID" "$MODEL_ROOT/hy-worldmirror" "HY-WorldMirror-2.0/*"
download_ms "qwen-image" "$QWEN_IMAGE_EDIT_MS_REPO_ID" "$MODEL_ROOT/Qwen-Image-Edit-2509"
download_ms "qwen3-vl" "$QWEN3_VL_MS_REPO_ID" "$MODEL_ROOT/Qwen3-VL-8B-Instruct"
download_ms "wan-i2v" "$WAN_I2V_MS_REPO_ID" "$MODEL_ROOT/Wan2.1-I2V-14B-720P-Diffusers"
download_ms "sam3" "$SAM3_MS_REPO_ID" "$MODEL_ROOT/sam3"
download_ms "dinov2" "$DINOV2_MS_REPO_ID" "$MODEL_ROOT/dinov2-base"
download_ms "grounding-dino" "$GROUNDING_DINO_MS_REPO_ID" "$MODEL_ROOT/grounding-dino-tiny"
download_ms "moge" "$MOGE_MS_REPO_ID" "$MODEL_ROOT/moge-2-vitl-normal"
# 默认管线使用非 DMD；DMD 仍下载便于回退
download_hf "worldstereo-memory" "$WORLDSTEREO_HF_REPO_ID" "$MODEL_ROOT/worldstereo" "worldstereo-memory/*"
download_hf "worldstereo-dmd" "$WORLDSTEREO_HF_REPO_ID" "$MODEL_ROOT/worldstereo" "worldstereo-memory-dmd/*"
download_hf "uni3c" "$UNI3C_HF_REPO_ID" "$MODEL_ROOT/Uni3C" "controlnet.pth"
download_ms "zim-anything" "$ZIM_ANYTHING_MS_REPO_ID" "$MODEL_ROOT/zim-anything-vitl"

echo
echo "== 路径校验 =="
MISSING_ITEMS=()
need_file "HY-Pano LoRA" "$MODEL_ROOT/HY-World-2.0-modelscope-full/HY-Pano-2.0/pytorch_lora_weights.safetensors"
need_file "WorldMirror config" "$MODEL_ROOT/hy-worldmirror/HY-WorldMirror-2.0/config.json"
need_file "WorldMirror weights" "$MODEL_ROOT/hy-worldmirror/HY-WorldMirror-2.0/model.safetensors"
need_file "Qwen-Image-Edit model_index" "$MODEL_ROOT/Qwen-Image-Edit-2509/model_index.json"
need_file "Qwen3-VL config" "$MODEL_ROOT/Qwen3-VL-8B-Instruct/config.json"
need_file "Wan I2V model_index" "$MODEL_ROOT/Wan2.1-I2V-14B-720P-Diffusers/model_index.json"
need_file "SAM3 config" "$MODEL_ROOT/sam3/config.json"
need_file "DINOv2 config" "$MODEL_ROOT/dinov2-base/config.json"
need_file "GroundingDINO config" "$MODEL_ROOT/grounding-dino-tiny/config.json"
need_file "MoGe model.pt" "$MODEL_ROOT/moge-2-vitl-normal/model.pt"
need_file "WorldStereo memory config" "$MODEL_ROOT/worldstereo/worldstereo-memory/config.json"
need_file "WorldStereo memory weights" "$MODEL_ROOT/worldstereo/worldstereo-memory/model.safetensors"
need_file "WorldStereo DMD config" "$MODEL_ROOT/worldstereo/worldstereo-memory-dmd/config.json"
need_file "WorldStereo DMD weights" "$MODEL_ROOT/worldstereo/worldstereo-memory-dmd/model.safetensors"
need_file "Uni3C controlnet.pth" "$MODEL_ROOT/Uni3C/controlnet.pth"

if [[ "$DRY_RUN" == 1 ]]; then
  echo
  echo "dry-run 完成，没有实际下载。"
  exit 0
fi

if [[ ${#MISSING_ITEMS[@]} -gt 0 ]]; then
  echo
  echo "仍有缺失项："
  printf '  - %s\n' "${MISSING_ITEMS[@]}"
  echo
  echo "说明：WorldStereo / Uni3C 默认走 HF 协议和 HF_ENDPOINT；如果镜像不可用，可设置 HF_ENDPOINT=https://huggingface.co 后重跑。"
fi

if [[ "$SKIP_PREFLIGHT" != 1 ]]; then
  echo
  echo "== 运行管线预检 =="
  source "$PIPELINE_ROOT/activate_env.sh"
  python "$PIPELINE_ROOT/python/preflight.py"
fi

echo
echo "模型下载流程结束。"
