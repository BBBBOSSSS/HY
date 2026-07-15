#!/usr/bin/env bash
# 用途：复现项目时下载/更新 HY-World-2.0 上游源码到管线默认读取路径。
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "$SETUP_DIR/.." && pwd)"
PIPELINES_ROOT="$(cd "$PIPELINE_ROOT/.." && pwd)"
HY2_BUNDLE_ROOT="${HY2_BUNDLE_ROOT:-$PIPELINES_ROOT/_bundle_deps}"

HY2_CODE_REPO_URL="${HY2_CODE_REPO_URL:-https://github.com/Tencent-Hunyuan/HY-World-2.0.git}"
HY2_CODE_ROOT="${HY2_CODE_ROOT:-$HY2_BUNDLE_ROOT/upstream/HY-World-2.0-gh}"
BRANCH="${BRANCH:-main}"
FORCE=0
DRY_RUN=0
SKIP_SUBMODULE=0

usage() {
  cat <<'EOF'
用法：
  bash setup_source_code.sh [选项]

常用选项：
  --repo URL          上游源码仓库，默认 https://github.com/Tencent-Hunyuan/HY-World-2.0.git
  --target PATH       源码目标路径，默认 /root/autodl-tmp/upstream/HY-World-2.0-gh
  --branch NAME       分支或 tag，默认 main
  --skip-submodule    不初始化 git submodule
  --force             如果目标目录已存在，删除后重新 clone
  --dry-run           只打印计划，不实际执行
  -h, --help          显示帮助

国内网络如果 GitHub 较慢，可以先设置自己的镜像地址：
  HY2_CODE_REPO_URL=https://你的镜像/HY-World-2.0.git bash setup_source_code.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) HY2_CODE_REPO_URL="$2"; shift 2 ;;
    --target) HY2_CODE_ROOT="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --skip-submodule) SKIP_SUBMODULE=1; shift ;;
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

echo "== HY2 上游源码准备 =="
echo "管线目录：$PIPELINE_ROOT"
echo "源码仓库：$HY2_CODE_REPO_URL"
echo "目标路径：$HY2_CODE_ROOT"
echo "分支/tag：$BRANCH"

if [[ -d "$HY2_CODE_ROOT/.git" && "$FORCE" != 1 ]]; then
  echo "[更新] 已存在 git 仓库，执行 fetch/checkout/pull"
  run_cmd git -C "$HY2_CODE_ROOT" fetch --all --tags
  run_cmd git -C "$HY2_CODE_ROOT" checkout "$BRANCH"
  run_cmd git -C "$HY2_CODE_ROOT" pull --ff-only
elif [[ -e "$HY2_CODE_ROOT" && "$FORCE" != 1 ]]; then
  echo "目标路径已存在但不是 git 仓库：$HY2_CODE_ROOT" >&2
  echo "如需覆盖，请加 --force" >&2
  exit 2
else
  if [[ -e "$HY2_CODE_ROOT" && "$FORCE" == 1 ]]; then
    run_cmd rm -rf "$HY2_CODE_ROOT"
  fi
  run_cmd mkdir -p "$(dirname "$HY2_CODE_ROOT")"
  run_cmd git clone --branch "$BRANCH" "$HY2_CODE_REPO_URL" "$HY2_CODE_ROOT"
fi

if [[ "$SKIP_SUBMODULE" != 1 ]]; then
  run_cmd git -C "$HY2_CODE_ROOT" submodule update --init --recursive
fi

if [[ "$DRY_RUN" != 1 ]]; then
  if [[ ! -f "$HY2_CODE_ROOT/hyworld2/worldgen/video_gen.py" ]]; then
    echo "[错误] 未找到 worldgen 代码：$HY2_CODE_ROOT/hyworld2/worldgen/video_gen.py" >&2
    exit 2
  fi
  echo "[OK] worldgen 代码已就绪：$HY2_CODE_ROOT/hyworld2/worldgen"
fi

echo
echo "源码准备完成。管线会默认读取："
echo "  HY2_CODE_ROOT=$HY2_CODE_ROOT"
