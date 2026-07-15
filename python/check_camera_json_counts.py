#!/usr/bin/env python3
# 用途：检查工具；验证轨迹 camera.json 的帧数是否匹配。
import argparse
import json
import os
from glob import glob


def main():
    parser = argparse.ArgumentParser(description="Check HY-World trajectory camera.json counts.")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    args = parser.parse_args()

    camera_paths = sorted(glob(os.path.join(args.scene_dir, "render_results", "**", "camera.json"), recursive=True))
    if not camera_paths:
        raise SystemExit(f"No camera.json files found under {args.scene_dir}/render_results")

    bad = []
    for camera_path in camera_paths:
        with open(camera_path) as f:
            camera = json.load(f)
        extrinsic_count = len(camera.get("extrinsic", []))
        intrinsic_count = len(camera.get("intrinsic", []))
        if extrinsic_count != args.expected_count or intrinsic_count != args.expected_count:
            rel = os.path.relpath(camera_path, args.scene_dir)
            bad.append((rel, extrinsic_count, intrinsic_count))

    print(f"checked {len(camera_paths)} camera.json files")
    if bad:
        print(f"bad camera counts, expected {args.expected_count}:")
        for rel, extrinsic_count, intrinsic_count in bad[:80]:
            print(f"  {rel}: extrinsic={extrinsic_count}, intrinsic={intrinsic_count}")
        raise SystemExit(1)

    print(f"all camera.json files have {args.expected_count} frames")


if __name__ == "__main__":
    main()
