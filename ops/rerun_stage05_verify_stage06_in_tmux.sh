#!/usr/bin/env bash
# Regenerate resolution-consistent GS data, verify it, then resume training.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="indoor_13c504b9a25f646eab0408efcd8835b9_candidate02"
RUN_DIR="/root/autodl-tmp/outputs/hy2_worldgen_runs/$NAME"
LOG="$ROOT/logs/stage05-resolution-stage06-20260719.log"

exec > >(tee -a "$LOG") 2>&1
source "$ROOT/scripts/env.sh"
source "$ROOT/scripts/activate_env.sh"

export CUDA_VISIBLE_DEVICES=0,1
echo "[$(date --iso-8601=seconds)] STAGE05_RESOLUTION_RETRY_START"
bash "$ROOT/stages/run_05_gs_data.sh" --name "$NAME" --nproc 2
stage05_rc=$?
echo "STAGE05_RESOLUTION_RETRY_RC=$stage05_rc"
[[ "$stage05_rc" -eq 0 ]] || exit "$stage05_rc"

python - <<PY
import json
from pathlib import Path
from PIL import Image

root = Path("$RUN_DIR/scene/gs_data")
image_paths = sorted((root / "images").glob("*.png"))
depth_paths = sorted((root / "depths").glob("*.png"))
normal_paths = sorted((root / "normals").glob("*.png"))
assert len(image_paths) == 661, len(image_paths)
assert len(depth_paths) == 310, len(depth_paths)
assert len(normal_paths) == 661, len(normal_paths)

mismatches = []
for depth_path in depth_paths:
    image_path = root / "images" / depth_path.name
    with Image.open(image_path) as image, Image.open(depth_path) as depth:
        if image.size != depth.size:
            mismatches.append((depth_path.name, image.size, depth.size))
assert not mismatches, mismatches[:10]

cameras = json.loads((root / "cameras.json").read_text())
sample = cameras["reconstruct_1-traj0_000003"]["intrinsic"]
assert abs(sample[0][2] - 416.0) < 1e-4, sample
assert abs(sample[1][2] - 240.0) < 1e-4, sample
print(
    f"GS_DATA_RESOLUTION_OK images={len(image_paths)} depths={len(depth_paths)} "
    f"normals={len(normal_paths)} size=832x480 sample_cxcy={sample[0][2]},{sample[1][2]}",
    flush=True,
)
PY
verify_rc=$?
echo "GS_DATA_RESOLUTION_VERIFY_RC=$verify_rc"
[[ "$verify_rc" -eq 0 ]] || exit "$verify_rc"

export CUDA_VISIBLE_DEVICES=0
echo "[$(date --iso-8601=seconds)] STAGE06_SINGLEGPU_AFTER_RESOLUTION_FIX_START"
bash "$ROOT/stages/run_06_train_gs.sh" --name "$NAME" --steps 8000 --nproc 1
stage06_rc=$?
echo "STAGE06_SINGLEGPU_AFTER_RESOLUTION_FIX_RC=$stage06_rc"
echo "[$(date --iso-8601=seconds)] STAGE05_06_RESOLUTION_REPAIR_RC=$stage06_rc"
exit "$stage06_rc"
