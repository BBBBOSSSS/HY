#!/usr/bin/env bash
# 用途：满血版一键入口；自动管理 vLLM，执行完整 VLM 链路，再训练导出 3DGS。
set -euo pipefail

PIPELINE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$PIPELINE_ROOT/scripts/env.sh"

NAME=""
IMAGE=""
PANO=""
NPROC="${NPROC_PER_NODE:-1}"
RENDER_NPROC="${HY2_RENDER_NPROC:-}"
NFRAME="${HY2_NFRAME:-21}"
STEPS=12000
PANO_NUM_CANDIDATES="${HY_PANO_NUM_CANDIDATES:-4}"
PROMPT=""
SCENE_TYPE="${HY2_SCENE_TYPE:-outdoor}"
FSDP=0
ALIGN_NFRAME="${HY2_ALIGN_NFRAME:-8}"
MAX_REFERENCE="${HY2_MAX_REFERENCE:-8}"
VLLM_SESSION="${HY2_VLLM_SESSION:-hy2_vllm}"
KEEP_VLLM=0
SKIP_PREFLIGHT=0
VLLM_START_TIMEOUT="${HY2_VLLM_START_TIMEOUT:-900}"
STAGE05_NPROC=""
STAGE06_NPROC=""

usage() {
  cat <<'EOF'
Usage:
  run_full_vlm_hy2.sh --name RUN_NAME (--image input.jpg | --panorama pano.png) [options]

Options:
  --scene-type indoor|outdoor
  --prompt TEXT
  --nproc N
  --render-nproc N       Stage 03 renderer processes; defaults to --nproc
  --nframe N
  --steps N
  --pano-num-candidates N
  --align-nframe N
  --max-reference N
  --fsdp
  --vllm-session NAME
  --keep-vllm
  --skip-preflight
  --vllm-start-timeout SECONDS
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --panorama) PANO="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --render-nproc) RENDER_NPROC="$2"; shift 2 ;;
    --nframe) NFRAME="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --pano-num-candidates) PANO_NUM_CANDIDATES="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --scene-type) SCENE_TYPE="$2"; shift 2 ;;
    --align-nframe) ALIGN_NFRAME="$2"; shift 2 ;;
    --max-reference) MAX_REFERENCE="$2"; shift 2 ;;
    --fsdp) FSDP=1; shift ;;
    --vllm-session) VLLM_SESSION="$2"; shift 2 ;;
    --keep-vllm) KEEP_VLLM=1; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    --vllm-start-timeout) VLLM_START_TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$NAME" ]] || { echo "--name required" >&2; usage >&2; exit 2; }
RENDER_NPROC="${RENDER_NPROC:-$NPROC}"
if [[ -n "$IMAGE" && -n "$PANO" ]]; then
  echo "Provide only one of --image or --panorama" >&2
  exit 2
fi
[[ -n "$IMAGE" || -n "$PANO" ]] || { echo "Provide --image or --panorama" >&2; usage >&2; exit 2; }

export HY2_VLLM_SESSION="$VLLM_SESSION"
source "$HY2_SCRIPT_ROOT/configure_gpu_topology.sh"
NPROC="${HY2_STAGE04_NPROC:-$NPROC}"
RENDER_NPROC="${HY2_RENDER_NPROC:-$RENDER_NPROC}"
STAGE05_NPROC="${HY2_STAGE05_NPROC:-$NPROC}"
STAGE06_NPROC="${HY2_STAGE06_NPROC:-$NPROC}"
# Stage02 只在 Qwen3-VL 物体标注时需要 vLLM；标注结束后立刻释放显存，
# 避免后续 SAM3 / WorldStereo 与 vLLM 抢显存。Stage03 需要视频 caption 时再重启。
export HY2_RELEASE_VLLM_CMD="${HY2_RELEASE_VLLM_CMD:-tmux kill-session -t ${VLLM_SESSION} 2>/dev/null || true}"
export HY2_RELEASE_VLLM_WAIT_S="${HY2_RELEASE_VLLM_WAIT_S:-10}"

cleanup_vllm() {
  if [[ "$KEEP_VLLM" == 1 ]]; then
    echo "[run_full_vlm_hy2] keep vLLM session: $VLLM_SESSION"
    return
  fi
  echo "[run_full_vlm_hy2] stopping vLLM session: $VLLM_SESSION"
  tmux kill-session -t "$VLLM_SESSION" 2>/dev/null || true
}

wait_for_vllm() {
  local deadline=$((SECONDS + VLLM_START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if python - "$HY2_LLM_ADDR" "$HY2_LLM_PORT" <<'PY'
import json
import sys
import urllib.request

host, port = sys.argv[1], int(sys.argv[2])
url = f"http://{host}:{port}/v1/models"
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("data"):
        raise SystemExit(0)
except Exception:
    raise SystemExit(1)
raise SystemExit(1)
PY
    then
      echo "[run_full_vlm_hy2] vLLM is ready at ${HY2_LLM_ADDR}:${HY2_LLM_PORT}"
      return 0
    fi

    if ! tmux has-session -t "$VLLM_SESSION" 2>/dev/null; then
      echo "[run_full_vlm_hy2] vLLM session exited before becoming ready: $VLLM_SESSION" >&2
      return 1
    fi
    echo "[run_full_vlm_hy2] waiting for vLLM at ${HY2_LLM_ADDR}:${HY2_LLM_PORT} ..."
    sleep 10
  done
  echo "[run_full_vlm_hy2] timed out waiting for vLLM after ${VLLM_START_TIMEOUT}s" >&2
  return 1
}

check_vlm_captions() {
  local scene_dir="$HY2_RUN_ROOT/$NAME/scene"
  python - "$scene_dir" <<'PY'
import json
import os
import sys
from glob import glob

scene_dir = sys.argv[1]
fallback = os.environ.get("WORLDGEN_FALLBACK_CAPTION", "")
videos = sorted(glob(os.path.join(scene_dir, "render_results", "*", "traj*", "render.mp4")))
missing = []
fallback_used = []
for video in videos:
    parts = video.split(os.sep)
    if parts[-3].startswith("reconstruct_") and parts[-2] == "traj1":
        continue
    caption_path = video.replace("/render.mp4", "/traj_caption.json")
    if not os.path.exists(caption_path):
        missing.append(os.path.relpath(video, scene_dir))
        continue
    try:
        prompt = json.load(open(caption_path, "r")).get("prompt", "")
    except Exception:
        missing.append(os.path.relpath(caption_path, scene_dir))
        continue
    if fallback and prompt.strip() == fallback.strip():
        fallback_used.append(os.path.relpath(caption_path, scene_dir))

if missing or fallback_used:
    if missing:
        print("Missing or invalid VLM captions:")
        for item in missing[:80]:
            print(f"  {item}")
    if fallback_used:
        print("Fallback captions found in full VLM mode:")
        for item in fallback_used[:80]:
            print(f"  {item}")
    raise SystemExit(1)
print(f"checked {len(videos)} rendered videos for full VLM captions")
PY
}

if [[ "$SKIP_PREFLIGHT" != 1 ]]; then
  source "$HY2_SCRIPT_ROOT/activate_env.sh"
  python "$HY2_PY_ROOT/preflight.py"
fi

if [[ -n "$PANO" ]]; then
  bash "$HY2_STAGE_ROOT/run_00_prepare_scene.sh" --name "$NAME" --panorama "$PANO" --scene-type "$SCENE_TYPE"
else
  bash "$HY2_STAGE_ROOT/run_00_prepare_scene.sh" --name "$NAME" --image "$IMAGE" --scene-type "$SCENE_TYPE"
  if [[ -n "$PROMPT" ]]; then
    bash "$HY2_STAGE_ROOT/run_01_panorama.sh" --name "$NAME" --prompt "$PROMPT" --num-candidates "$PANO_NUM_CANDIDATES"
  else
    bash "$HY2_STAGE_ROOT/run_01_panorama.sh" --name "$NAME" --num-candidates "$PANO_NUM_CANDIDATES"
  fi
fi

tmux kill-session -t "$VLLM_SESSION" 2>/dev/null || true
bash "$HY2_SCRIPT_ROOT/start_vllm_qwen3vl.sh" "$VLLM_SESSION"
trap cleanup_vllm EXIT
wait_for_vllm

CUDA_VISIBLE_DEVICES="$HY2_STAGE02_CUDA_VISIBLE_DEVICES" bash "$HY2_STAGE_ROOT/run_02_worldnav.sh" --name "$NAME" --mode vlm-full --nframe "$NFRAME"

tmux kill-session -t "$VLLM_SESSION" 2>/dev/null || true
bash "$HY2_SCRIPT_ROOT/start_vllm_qwen3vl.sh" "$VLLM_SESSION"
wait_for_vllm

bash "$HY2_STAGE_ROOT/run_03_traj_render.sh" --name "$NAME" --mode vlm-full --nproc "$RENDER_NPROC" --expected-nframe "$NFRAME"
check_vlm_captions
cleanup_vllm
trap - EXIT

ws_args=(--name "$NAME" --nproc "$NPROC" --align-nframe "$ALIGN_NFRAME" --max-reference "$MAX_REFERENCE")
[[ "$FSDP" == 1 ]] && ws_args+=(--fsdp)
bash "$HY2_STAGE_ROOT/run_04_worldstereo.sh" "${ws_args[@]}"
CUDA_VISIBLE_DEVICES="$HY2_STAGE05_CUDA_VISIBLE_DEVICES" bash "$HY2_STAGE_ROOT/run_05_gs_data.sh" --name "$NAME" --nproc "$STAGE05_NPROC"
CUDA_VISIBLE_DEVICES="$HY2_STAGE06_CUDA_VISIBLE_DEVICES" bash "$HY2_STAGE_ROOT/run_06_train_gs.sh" --name "$NAME" --steps "$STEPS" --nproc "$STAGE06_NPROC"

echo "DONE: $HY2_RUN_ROOT/$NAME"
