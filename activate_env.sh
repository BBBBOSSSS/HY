#!/usr/bin/env bash
# 用途：根目录环境激活入口；加载 scripts/activate_env.sh，激活 HY2 Python/CUDA 环境。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/activate_env.sh"
