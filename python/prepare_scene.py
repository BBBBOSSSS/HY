#!/usr/bin/env python3
# 用途：阶段 00 工具；复制并校验输入，写入场景元信息。
import argparse
import json
import shutil
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--run-root", required=True)
parser.add_argument("--image")
parser.add_argument("--panorama")
parser.add_argument("--scene-type", default="outdoor", choices=["indoor", "outdoor"])
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

run_dir = Path(args.run_root) / args.name
scene_dir = run_dir / "scene"
scene_dir.mkdir(parents=True, exist_ok=True)

if args.panorama:
    src = Path(args.panorama)
    if not src.exists():
        raise FileNotFoundError(src)
    dst = scene_dir / "panorama.png"
    if args.overwrite or not dst.exists():
        img = Image.open(src).convert("RGB")
        if img.size[0] != img.size[1] * 2:
            raise SystemExit(
                f"Panorama must be standard 2:1 equirectangular, got size={img.size}. "
                "Please regenerate with the fixed panorama stage."
            )
        img.save(dst)
    print(dst)
elif args.image:
    src = Path(args.image)
    if not src.exists():
        raise FileNotFoundError(src)
    dst = scene_dir / "input.png"
    if args.overwrite or not dst.exists():
        Image.open(src).convert("RGB").save(dst)
    print(dst)
else:
    raise SystemExit("Provide --image or --panorama")

meta = scene_dir / "meta_info.json"
if args.overwrite or not meta.exists():
    meta.write_text(json.dumps({"scene_type": args.scene_type}, indent=2), encoding="utf-8")

(run_dir / "logs").mkdir(exist_ok=True)
print(f"RUN_DIR={run_dir}")
print(f"SCENE_DIR={scene_dir}")
