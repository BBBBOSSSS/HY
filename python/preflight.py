#!/usr/bin/env python3
# 用途：预检工具；检查本地模型、代码路径、CUDA 和 gsplat 是否就绪。
import os
import json
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

_moge_raw = Path(os.environ.get("MOGE_MODEL_ID", "/root/autodl-tmp/models/moge-2-vitl-normal"))
# MoGeModel.from_pretrained accepts a .pt file directly; if the env var
# already points to a .pt file, check it as-is; otherwise treat it as a
# directory containing model.pt.
if _moge_raw.suffix == ".pt":
    _moge_check = _moge_raw
else:
    _moge_check = _moge_raw / "model.pt"
MIN_SIZES = {
    "MoGe model.pt": (_moge_check, 500 * 1024 * 1024),
    "Uni3C controlnet.pth": (Path(os.environ.get("UNI3C_CONTROLNET_PATH", "/root/autodl-tmp/models/Uni3C/controlnet.pth")), 1024 * 1024 * 1024),
}

pano_backend = os.environ.get("HY_PANO_BACKEND", "auto")
hy_pano_model_path = Path(os.environ.get("HY_PANO_MODEL_PATH", "/root/autodl-tmp/models/HY-World-2.0-modelscope-full/HY-Pano-2.0"))
hy_pano_index = hy_pano_model_path / "model.safetensors.index.json"
qwen_pano_base = Path(os.environ.get("QWEN_PANO_BASE", "/root/autodl-tmp/models/Qwen-Image-Edit-2509"))
hy_pano_lora = Path(os.environ.get("HY_PANO_LORA_REPO", "/root/autodl-tmp/models/HY-World-2.0-modelscope-full")) / os.environ.get("HY_PANO_LORA_SUBFOLDER", "HY-Pano-2.0") / "pytorch_lora_weights.safetensors"
has_hy_pano_full = hy_pano_index.exists()


def qwen_pano_required_paths(base_path: Path):
    return [
        base_path / "model_index.json",
        base_path / "configuration.json",
        base_path / "transformer" / "config.json",
        base_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
        base_path / "text_encoder" / "config.json",
        base_path / "text_encoder" / "model.safetensors.index.json",
        base_path / "vae" / "config.json",
        base_path / "vae" / "diffusion_pytorch_model.safetensors",
        base_path / "processor" / "tokenizer.json",
    ]


def qwen_pano_ready(base_path: Path) -> bool:
    return all(path.exists() for path in qwen_pano_required_paths(base_path))

CHECKS = [
    ("HY2 code root", Path(os.environ["HY2_CODE_ROOT"])),
    ("worldgen code", Path(os.environ["HY2_WORLDGEN_ROOT"]) / "video_gen.py"),
    ("WorldStereo root", Path(os.environ["WORLDSTEREO_REPO_ID"])),
    # 默认走非 DMD；DMD 权重改为可选（见 OPTIONAL）
    ("WorldStereo memory config", Path(os.environ["WORLDSTEREO_REPO_ID"]) / "worldstereo-memory" / "config.json"),
    ("WorldStereo memory weights", Path(os.environ["WORLDSTEREO_REPO_ID"]) / "worldstereo-memory" / "model.safetensors"),
    ("Wan base", Path(os.environ["WORLDSTEREO_BASE_MODEL_PATH"])),
    ("SAM3", Path(os.environ["SAM3_REPO_ID"])),
    ("Qwen3-VL", Path(os.environ["QWEN3_VL_MODEL_PATH"])),
]

if pano_backend == "hy-pano2" or (pano_backend == "auto" and has_hy_pano_full):
    CHECKS.extend([
        ("HY-Pano backend", hy_pano_model_path),
        ("HY-Pano full weights", hy_pano_index),
    ])
else:
    CHECKS.extend([
        ("Qwen pano base", qwen_pano_base),
        ("HY-Pano LoRA", hy_pano_lora),
    ])

OPTIONAL = [
    ("ZIM anything local cache", os.environ.get("NAVER_IV_ZIM_ANYTHING_VITL_PATH")),
    ("GroundingDINO local cache", os.environ.get("IDEA_RESEARCH_GROUNDING_DINO_TINY_PATH")),
    (
        "WorldStereo DMD config (optional)",
        str(Path(os.environ["WORLDSTEREO_REPO_ID"]) / "worldstereo-memory-dmd" / "config.json"),
    ),
    (
        "WorldStereo DMD weights (optional)",
        str(Path(os.environ["WORLDSTEREO_REPO_ID"]) / "worldstereo-memory-dmd" / "model.safetensors"),
    ),
]

def size(path: Path) -> str:
    try:
        return subprocess.check_output(["du", "-sh", str(path)], text=True, stderr=subprocess.DEVNULL).split()[0]
    except Exception:
        return "?"

def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"[unavailable] {e}"

def compiled_cuda_arches(path: Path):
    if not path.exists():
        return []
    out = run(["cuobjdump", "--list-elf", str(path)])
    arches = []
    for token in out.replace("\n", " ").split():
        if ".sm_" in token:
            arch = token.split(".sm_", 1)[1].split(".", 1)[0]
            arches.append(arch)
    return sorted(set(arches))

def torch_info():
    code = r"""
import json
import sys
try:
    import torch
    info = {
        "ok": True,
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            cap = torch.cuda.get_device_capability(idx)
            info["devices"].append({
                "idx": idx,
                "name": torch.cuda.get_device_name(idx),
                "capability": [cap[0], cap[1]],
                "memory_gib": props.total_memory / 1024**3,
            })
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    print(json.dumps(info))
except Exception as e:
    print(json.dumps({"ok": False, "error": repr(e), "python": sys.executable}))
"""
    try:
        out = subprocess.check_output([sys.executable, "-c", code], text=True, stderr=subprocess.STDOUT, timeout=60)
        return json.loads(out.strip().splitlines()[-1])
    except Exception as e:
        return {"ok": False, "error": repr(e), "python": sys.executable}

missing = []
print("== HY2.0 official route preflight ==")
for name, path in CHECKS:
    ok = path.exists()
    print(("[OK] " if ok else "[MISSING] ") + f"{name}: {path}" + (f" ({size(path)})" if ok else ""))
    if not ok:
        missing.append(name)

if pano_backend != "hy-pano2" and not has_hy_pano_full:
    qwen_missing = [path for path in qwen_pano_required_paths(qwen_pano_base) if not path.exists()]
    if qwen_missing:
        print("[MISSING/INCOMPLETE] Qwen pano base required files:")
        for path in qwen_missing:
            print("-", path)
        missing.append("Qwen pano base required files")

for name, (path, min_bytes) in MIN_SIZES.items():
    ok = path.exists() and path.stat().st_size >= min_bytes
    detail = f"{path}" + (f" ({size(path)})" if path.exists() else "")
    print(("[OK] " if ok else "[MISSING/INCOMPLETE] ") + f"{name}: {detail}")
    if not ok:
        missing.append(name)

print("\n== optional aux checkpoints ==")
for name, path in OPTIONAL:
    if path:
        p = Path(path)
        print(("[OK] " if p.exists() else "[MISSING] ") + f"{name}: {p}")
    else:
        print(f"[not set] {name}: official code may download/use it for outdoor sky segmentation if HY2_OFFLINE_AUX=0")

print("\n== GPU runtime ==")
info = torch_info()
print(f"Python: {info.get('python', sys.executable)}")
if info.get("ok"):
    print(f"Torch: {info['torch']}; torch_cuda={info['torch_cuda']}; cuda_available={info['cuda_available']}; gpus={info['device_count']}")
else:
    print(f"[WARN] torch probe failed: {info.get('error')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', '(not set)')}")
nvcc_out = run(["nvcc", "--version"])
print(f"nvcc: {nvcc_out.splitlines()[-1] if 'release' in nvcc_out else nvcc_out}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")

require_cuda = os.environ.get("HY2_REQUIRE_CUDA", "0") == "1"
expected_gpu_regex = os.environ.get("HY2_EXPECTED_GPU_REGEX")
if info.get("ok") and info.get("cuda_available"):
    import re

    for dev in info["devices"]:
        cap = dev["capability"]
        print(f"[OK] GPU {dev['idx']}: {dev['name']}; capability={cap[0]}.{cap[1]}; memory={dev['memory_gib']:.1f} GiB; bf16={info.get('bf16_supported')}")
        if expected_gpu_regex and not re.search(expected_gpu_regex, dev["name"], re.IGNORECASE):
            print(f"[WARN] GPU {dev['idx']} name does not match HY2_EXPECTED_GPU_REGEX={expected_gpu_regex!r}")

    gsplat_so = Path(os.environ.get("HY2_GSPLAT_ROOT", Path(os.environ["HY2_WORLDGEN_ROOT"]) / "third_party/gsplat_maskgaussian")) / "gsplat" / "csrc.so"
    arches = compiled_cuda_arches(gsplat_so)
    first_cap = info["devices"][0]["capability"]
    current = f"{first_cap[0]}{first_cap[1]}"
    if arches:
        print(f"gsplat csrc arches: {', '.join('sm_' + a for a in arches)}")
        if current not in arches:
            print(f"[WARN] gsplat extension lacks sm_{current}; rebuild it before 3DGS training/viewer.")
            print(f"       cd {gsplat_so.parents[1]} && TORCH_CUDA_ARCH_LIST='{first_cap[0]}.{first_cap[1]}' pip install -e . --no-build-isolation")
    else:
        print(f"[WARN] could not inspect gsplat csrc arches at {gsplat_so}")
elif require_cuda:
    missing.append("CUDA GPU runtime")
    print("[MISSING] HY2_REQUIRE_CUDA=1 but torch.cuda.is_available() is False")

if missing:
    print("\nBlocking missing/incomplete items:")
    for item in missing:
        print("-", item)
    raise SystemExit(2)
print("\nPreflight passed.")
