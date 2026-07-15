#!/usr/bin/env bash
# 用途：阶段 01 启动入口；从输入视角图生成全景候选，并选择最终全景图。
set -euo pipefail
STAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$STAGE_DIR/../scripts/env.sh"
source "$HY2_SCRIPT_ROOT/activate_env.sh"
NAME=""; PROMPT=""; NEGATIVE_PROMPT=""; STEPS=40; SEED=42; SEED_STEP=97; NUM_CANDIDATES=4; OVERWRITE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --negative-prompt) NEGATIVE_PROMPT="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --seed-step) SEED_STEP="$2"; shift 2 ;;
    --num-candidates) NUM_CANDIDATES="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$NAME" ]] || { echo "--name required" >&2; exit 2; }
SCENE_DIR="$HY2_RUN_ROOT/$NAME/scene"
INPUT="$SCENE_DIR/input.png"
PANO="$SCENE_DIR/panorama.png"
SCENE_TYPE="$(python - "$SCENE_DIR/meta_info.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
scene_type = "indoor"
if path.exists():
    try:
        scene_type = json.loads(path.read_text()).get("scene_type", scene_type)
    except Exception:
        pass
print(scene_type if scene_type in {"indoor", "outdoor"} else "indoor")
PY
)"
[[ -f "$INPUT" ]] || { echo "missing $INPUT; run run_00_prepare_scene.sh --image first" >&2; exit 2; }
if [[ -f "$PANO" && "$OVERWRITE" != 1 ]]; then
  echo "exists: $PANO"
  exit 0
fi
python "$HY2_PY_ROOT/require_cuda.py" --stage "stage01 panorama"
cmd=(python "$HY2_PY_ROOT/generate_and_select_panorama.py" --image "$INPUT" --save "$PANO" --scene-type "$SCENE_TYPE" --backend "$HY_PANO_BACKEND" --hy-pano-model "$HY_PANO_MODEL_PATH" --model-id "$QWEN_PANO_BASE" --lora-repo "$HY_PANO_LORA_REPO" --lora-subfolder "$HY_PANO_LORA_SUBFOLDER" --device-map "$HY_PANO_DEVICE_MAP" --max-gpu-memory "$HY_PANO_MAX_GPU_MEMORY" --max-cpu-memory "$HY_PANO_MAX_CPU_MEMORY" --offload-dir "$HY_PANO_OFFLOAD_DIR" --true-cfg-scale "$HY_PANO_TRUE_CFG_SCALE" --steps "$STEPS" --seed "$SEED" --seed-step "$SEED_STEP" --num-candidates "$NUM_CANDIDATES")
if [[ -n "$PROMPT" ]]; then cmd+=(--prompt "$PROMPT"); fi
if [[ -n "$NEGATIVE_PROMPT" ]]; then cmd+=(--negative-prompt "$NEGATIVE_PROMPT"); fi
"${cmd[@]}" 2>&1 | tee "$HY2_RUN_ROOT/$NAME/logs/01_panorama.log"
