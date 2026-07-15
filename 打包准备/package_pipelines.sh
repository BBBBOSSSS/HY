#!/usr/bin/env bash
# 用途：把 /root/autodl-tmp/pipelines 打成可迁移归档包；建议输出到外部盘或更大磁盘。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINES_ROOT="$(cd "$PIPELINE_ROOT/.." && pwd)"
OUT=""
DRY_RUN=0

usage() {
  cat <<'EOF'
用法：
  bash package_pipelines.sh --out /path/to/pipelines_hy2_bundle.tar
  bash package_pipelines.sh --out /path/to/pipelines_hy2_bundle.tar.zst

选项：
  --out PATH   输出 tar 文件路径，建议放到外部盘或剩余空间足够的位置
  --dry-run    只打印计划，不真正打包
  -h, --help   显示帮助

注意：
  默认 _bundle_deps 不包含模型权重，只包含源码和 conda 环境。
  如果你手动用 --include-models 收进模型，归档逻辑体积可能超过 250G。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$OUT" ]] || { echo "--out required" >&2; usage >&2; exit 2; }

echo "== HY2 pipelines 打包 =="
echo "源目录：$PIPELINES_ROOT"
echo "输出：$OUT"
echo "逻辑体积："
du -sh "$PIPELINES_ROOT" 2>/dev/null || true

if [[ "$OUT" == /root/autodl-tmp/* ]]; then
  echo
  echo "[警告] 输出路径在 /root/autodl-tmp 当前盘内。conda 环境较大，可能空间不足。"
  df -h /root/autodl-tmp
fi

if [[ "$DRY_RUN" == 1 ]]; then
  echo
  echo "dry-run 完成，没有实际打包。"
  exit 0
fi

mkdir -p "$(dirname "$OUT")"
if [[ "$OUT" == *.zst ]]; then
  tar --hard-dereference --exclude 'pipelines/_bundle_deps/models' -I "zstd -T1 -1" -cf "$OUT" -C "$(dirname "$PIPELINES_ROOT")" "$(basename "$PIPELINES_ROOT")"
else
  tar --hard-dereference --exclude 'pipelines/_bundle_deps/models' -cf "$OUT" -C "$(dirname "$PIPELINES_ROOT")" "$(basename "$PIPELINES_ROOT")"
fi

echo
echo "打包完成：$OUT"
du -sh "$OUT" 2>/dev/null || true
