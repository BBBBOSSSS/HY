#!/usr/bin/env bash
# 用途：根目录满血一键入口；转发到 stages/run_full_vlm_hy2.sh，自动管理 vLLM 并运行全链路。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/stages/run_full_vlm_hy2.sh" "$@"
