#!/usr/bin/env bash
# 用途：根目录环境入口；加载 scripts/env.sh，导出管线路径、模型路径和运行默认值。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/env.sh"
