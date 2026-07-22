#!/usr/bin/env bash
# Resume the candidate02 full pipeline after WorldStereo/WorldMirror completed.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="indoor_13c504b9a25f646eab0408efcd8835b9_candidate02"
LOG="$ROOT/logs/stage05-06-repair-20260719.log"

exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] STAGE05_06_REPAIR_START"
source "$ROOT/scripts/env.sh"
source "$ROOT/scripts/activate_env.sh"
export CUDA_VISIBLE_DEVICES=0,1

python - <<'PY'
import os
from moge.model.v2 import MoGeModel

path = os.environ["MOGE_MODEL_ID"]
model = MoGeModel.from_pretrained(path)
print(f"MOGE_LOCAL_OK={path} params={sum(p.numel() for p in model.parameters())}", flush=True)
PY
model_rc=$?
echo "MOGE_LOCAL_RC=$model_rc"
if [[ "$model_rc" -ne 0 ]]; then
  echo "[$(date --iso-8601=seconds)] STAGE05_06_REPAIR_RC=$model_rc"
  exit "$model_rc"
fi

echo "[$(date --iso-8601=seconds)] STAGE05_DUALGPU_RETRY_START"
bash "$ROOT/stages/run_05_gs_data.sh" --name "$NAME" --nproc 2
stage05_rc=$?
echo "STAGE05_DUALGPU_RETRY_RC=$stage05_rc"
if [[ "$stage05_rc" -ne 0 ]]; then
  echo "[$(date --iso-8601=seconds)] STAGE05_06_REPAIR_RC=$stage05_rc"
  exit "$stage05_rc"
fi

echo "[$(date --iso-8601=seconds)] STAGE06_DUALGPU_START"
bash "$ROOT/stages/run_06_train_gs.sh" --name "$NAME" --steps 12000 --nproc 2
stage06_rc=$?
echo "STAGE06_DUALGPU_RC=$stage06_rc"
echo "[$(date --iso-8601=seconds)] STAGE05_06_REPAIR_RC=$stage06_rc"
exit "$stage06_rc"
