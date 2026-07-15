#!/usr/bin/env bash
# 用途：把 HY2 运行依赖收进 pipelines/_bundle_deps，默认用硬链接避免重复占用磁盘。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINES_ROOT="$(cd "$PIPELINE_ROOT/.." && pwd)"
BUNDLE_ROOT="${HY2_BUNDLE_ROOT:-$PIPELINES_ROOT/_bundle_deps}"
MODE="link"
FORCE=0
DRY_RUN=0
INCLUDE_MODELS=0

usage() {
  cat <<'EOF'
用法：
  bash prepare_bundle_deps.sh [选项]

默认行为：
  把 /root/autodl-tmp/upstream、/root/autodl-tmp/conda-envs 中本管线需要的内容，
  用硬链接收进 /root/autodl-tmp/pipelines/_bundle_deps。
  模型权重默认不收进去，因为 模型下载/model_download.sh 已经负责下载。

选项：
  --include-models  连模型权重也收进 _bundle_deps/models；体积会超过 200G
  --copy       真实复制文件；需要额外数百 GB 空间，不推荐在当前机器上用
  --force      如果目标已存在，先删除后重新收集
  --dry-run    只打印计划，不执行
  -h, --help   显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --copy) MODE="copy"; shift ;;
    --include-models) INCLUDE_MODELS=1; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" != 1 ]]; then
    "$@"
  fi
}

collect_dir() {
  local src="$1"
  local dst="$2"
  local label="$3"

  if [[ ! -e "$src" ]]; then
    echo "[跳过] $label：源不存在 $src"
    return
  fi
  if [[ -e "$dst" && "$FORCE" != 1 ]]; then
    echo "[保留] $label 已存在：$dst"
    return
  fi
  if [[ -e "$dst" && "$FORCE" == 1 ]]; then
    run_cmd rm -rf "$dst"
  fi

  run_cmd mkdir -p "$(dirname "$dst")"
  if [[ "$MODE" == "copy" ]]; then
    run_cmd cp -a "$src" "$dst"
  else
    run_cmd cp -al "$src" "$dst"
  fi
}

echo "== HY2 打包依赖收集 =="
echo "pipelines 目录：$PIPELINES_ROOT"
echo "bundle 目录：$BUNDLE_ROOT"
echo "模式：$MODE"

if [[ "$INCLUDE_MODELS" == 1 ]]; then
  collect_dir /root/autodl-tmp/models "$BUNDLE_ROOT/models" "模型权重"
else
  echo "[跳过] 模型权重：默认不放进包里，请用 模型下载/model_download.sh 在目标机器下载。"
fi
collect_dir /root/autodl-tmp/upstream/HY-World-2.0-gh "$BUNDLE_ROOT/upstream/HY-World-2.0-gh" "HY-World-2.0 上游源码"
collect_dir /root/autodl-tmp/upstream/moge_src "$BUNDLE_ROOT/upstream/moge_src" "MoGe 源码"
collect_dir /root/autodl-tmp/conda-envs/hyworld2 "$BUNDLE_ROOT/conda-envs/hyworld2" "HY2 conda 环境"
collect_dir /root/autodl-tmp/conda-envs/vllm_qwen "$BUNDLE_ROOT/conda-envs/vllm_qwen" "vLLM conda 环境"

echo
echo "依赖收集完成。"
echo "逻辑体积："
du -sh "$BUNDLE_ROOT" 2>/dev/null || true
echo
echo "说明：默认硬链接不会额外占用同等磁盘空间；如果之后 tar 到外部盘，包里会包含这些依赖内容。"
