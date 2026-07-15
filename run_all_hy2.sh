#!/usr/bin/env bash
# 用途：根目录低层全链路入口；转发到 stages/run_all_hy2.sh，不负责管理 vLLM 生命周期。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/stages/run_all_hy2.sh" "$@"
