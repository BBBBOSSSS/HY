#!/usr/bin/env bash
# Download the official torchvision VGG16 weights without a proxy, verify them,
# then resume the upstream-recommended one-GPU training path.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="indoor_13c504b9a25f646eab0408efcd8835b9_candidate02"
LOG="$ROOT/logs/vgg16-stage06-repair-20260719.log"
CACHE_DIR="/root/.cache/torch/hub/checkpoints"
FINAL="$CACHE_DIR/vgg16-397923af.pth"
PART_DIR="/root/autodl-tmp/_bundle_deps/vgg16-397923af.parts"
ASSEMBLED="$PART_DIR/vgg16-397923af.pth.assembled"
URL="https://download.pytorch.org/models/vgg16-397923af.pth"
TOTAL=553433881
PARTS=8

exec > >(tee -a "$LOG") 2>&1
echo "[$(date --iso-8601=seconds)] VGG16_DIRECT_REPAIR_START"
mkdir -p "$CACHE_DIR" "$PART_DIR"

valid_final=0
if [[ -f "$FINAL" ]]; then
  final_hash=$(sha256sum "$FINAL" | awk '{print $1}')
  [[ "$final_hash" == 397923af* ]] && valid_final=1
fi

if [[ "$valid_final" -ne 1 ]]; then
  pids=()
  chunk=$(((TOTAL + PARTS - 1) / PARTS))
  for ((i=0; i<PARTS; i++)); do
    start=$((i * chunk))
    end=$((start + chunk - 1))
    (( end >= TOTAL )) && end=$((TOTAL - 1))
    part=$(printf '%s/part_%02d' "$PART_DIR" "$i")
    echo "VGG16_PART_START index=$i range=$start-$end"
    curl --noproxy '*' --fail --location --silent --show-error \
      --retry 8 --retry-delay 2 --retry-all-errors \
      --range "$start-$end" --output "$part" "$URL" &
    pids+=("$!")
  done

  while true; do
    active=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then active=$((active + 1)); fi
    done
    downloaded=$(find "$PART_DIR" -maxdepth 1 -type f -name 'part_*' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')
    echo "VGG16_DIRECT_PROGRESS bytes=$downloaded/$TOTAL active_parts=$active"
    [[ "$active" -eq 0 ]] && break
    sleep 10
  done

  download_rc=0
  for pid in "${pids[@]}"; do
    wait "$pid" || download_rc=1
  done
  echo "VGG16_DIRECT_DOWNLOAD_RC=$download_rc"
  [[ "$download_rc" -eq 0 ]] || exit "$download_rc"

  for ((i=0; i<PARTS; i++)); do
    start=$((i * chunk))
    end=$((start + chunk - 1))
    (( end >= TOTAL )) && end=$((TOTAL - 1))
    expected=$((end - start + 1))
    part=$(printf '%s/part_%02d' "$PART_DIR" "$i")
    actual=$(stat -c '%s' "$part")
    if [[ "$actual" -ne "$expected" ]]; then
      echo "VGG16_PART_SIZE_ERROR index=$i expected=$expected actual=$actual" >&2
      exit 1
    fi
  done

  cp "$PART_DIR/part_00" "$ASSEMBLED"
  for ((i=1; i<PARTS; i++)); do
    part=$(printf '%s/part_%02d' "$PART_DIR" "$i")
    dd if="$part" of="$ASSEMBLED" oflag=append conv=notrunc status=none
  done
  assembled_size=$(stat -c '%s' "$ASSEMBLED")
  assembled_hash=$(sha256sum "$ASSEMBLED" | awk '{print $1}')
  echo "VGG16_ASSEMBLED size=$assembled_size sha256=$assembled_hash"
  [[ "$assembled_size" -eq "$TOTAL" && "$assembled_hash" == 397923af* ]] || exit 1
  mv "$ASSEMBLED" "$FINAL"
fi

echo "VGG16_LOCAL_OK=$(sha256sum "$FINAL" | awk '{print $1}')"
source "$ROOT/scripts/env.sh"
source "$ROOT/scripts/activate_env.sh"
export CUDA_VISIBLE_DEVICES=0

python - <<'PY'
import torch
x = torch.ones(1_000_000, device="cuda")
assert x.sum().item() == 1_000_000
torch.cuda.synchronize()
print(f"CUDA_SINGLE_OK={torch.cuda.get_device_name(0)}", flush=True)
PY
cuda_rc=$?
echo "CUDA_SINGLE_RC=$cuda_rc"
[[ "$cuda_rc" -eq 0 ]] || exit "$cuda_rc"

echo "[$(date --iso-8601=seconds)] STAGE06_SINGLEGPU_RETRY_START"
bash "$ROOT/stages/run_06_train_gs.sh" --name "$NAME" --steps 8000 --nproc 1
stage06_rc=$?
echo "STAGE06_SINGLEGPU_RETRY_RC=$stage06_rc"
echo "[$(date --iso-8601=seconds)] VGG16_STAGE06_REPAIR_RC=$stage06_rc"
exit "$stage06_rc"
