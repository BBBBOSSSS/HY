#!/usr/bin/env bash
# Fallback for hosts where two-GPU NCCL initialization hits a CUDA fault.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
NAME="indoor_13c504b9a25f646eab0408efcd8835b9_candidate02"
LOG="$ROOT/logs/stage06-single-repair-20260719.log"

exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] STAGE06_SINGLE_REPAIR_START"
source "$ROOT/scripts/env.sh"
source "$ROOT/scripts/activate_env.sh"
export CUDA_VISIBLE_DEVICES=0

python - <<'PY'
import torch

assert torch.cuda.is_available()
x = torch.arange(1_000_000, device="cuda", dtype=torch.float32)
y = (x.square().mean()).item()
torch.cuda.synchronize()
print(f"CUDA_SINGLE_OK={torch.cuda.get_device_name(0)} value={y:.1f}", flush=True)
PY
cuda_rc=$?
echo "CUDA_SINGLE_RC=$cuda_rc"
if [[ "$cuda_rc" -ne 0 ]]; then
  echo "[$(date --iso-8601=seconds)] STAGE06_SINGLE_REPAIR_RC=$cuda_rc"
  exit "$cuda_rc"
fi

# The upstream guide recommends 8,000 steps for one GPU.
echo "[$(date --iso-8601=seconds)] STAGE06_SINGLEGPU_START"
bash "$ROOT/stages/run_06_train_gs.sh" --name "$NAME" --steps 8000 --nproc 1
stage06_rc=$?
echo "STAGE06_SINGLEGPU_RC=$stage06_rc"
echo "[$(date --iso-8601=seconds)] STAGE06_SINGLE_REPAIR_RC=$stage06_rc"
exit "$stage06_rc"
