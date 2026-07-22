#!/usr/bin/env bash
# Install the upstream-declared TensorBoard dependency and resume GS training.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
NAME="indoor_13c504b9a25f646eab0408efcd8835b9_candidate02"
LOG="$ROOT/logs/stage06-repair-20260719.log"

exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] TENSORBOARD_REPAIR_START"
source "$ROOT/scripts/env.sh"
source "$ROOT/scripts/activate_env.sh"

python -m pip install tensorboard
install_rc=$?
echo "TENSORBOARD_INSTALL_RC=$install_rc"
if [[ "$install_rc" -ne 0 ]]; then
  echo "[$(date --iso-8601=seconds)] STAGE06_REPAIR_RC=$install_rc"
  exit "$install_rc"
fi

python - <<'PY'
import tensorboard
from torch.utils.tensorboard import SummaryWriter
print(f"TENSORBOARD_OK={tensorboard.__version__} writer={SummaryWriter.__name__}", flush=True)
PY
import_rc=$?
echo "TENSORBOARD_IMPORT_RC=$import_rc"
if [[ "$import_rc" -ne 0 ]]; then
  echo "[$(date --iso-8601=seconds)] STAGE06_REPAIR_RC=$import_rc"
  exit "$import_rc"
fi

export CUDA_VISIBLE_DEVICES=0,1
echo "[$(date --iso-8601=seconds)] STAGE06_DUALGPU_RETRY_START"
bash "$ROOT/stages/run_06_train_gs.sh" --name "$NAME" --steps 12000 --nproc 2
stage06_rc=$?
echo "STAGE06_DUALGPU_RETRY_RC=$stage06_rc"
echo "[$(date --iso-8601=seconds)] STAGE06_REPAIR_RC=$stage06_rc"
exit "$stage06_rc"
