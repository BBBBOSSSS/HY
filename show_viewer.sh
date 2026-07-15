#!/usr/bin/env bash
# 用途：根目录 viewer 入口；转发到 stages/show_viewer.sh，打开官方 GS 查看器。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/stages/show_viewer.sh" "$@"
