#!/usr/bin/env python3
# 用途：阶段 03 工具；使用 Qwen3-VL 重新描述已渲染的轨迹视频。
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--llm-addr", default="127.0.0.1")
    parser.add_argument("--llm-port", type=int, default=8000)
    parser.add_argument("--llm-name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    videos = sorted(glob(os.path.join(args.scene_dir, "render_results", "*", "traj*", "render.mp4")))
    caption_videos = [
        path for path in videos
        if not (path.split(os.sep)[-3].startswith("reconstruct_") and path.split(os.sep)[-2] == "traj1")
    ]
    if not videos:
        raise SystemExit(f"no render.mp4 files under {args.scene_dir}")

    from src.vlm_utils import get_traj_caption

    def recaption(path):
        caption = get_traj_caption(args.llm_addr, args.llm_port, args.llm_name, path)
        output_path = path.replace("/render.mp4", "/traj_caption.json")
        with open(output_path, "w") as f:
            json.dump({"prompt": caption}, f, indent=2)
        return output_path

    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(recaption, path) for path in caption_videos]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="recaption"):
            try:
                fut.result()
            except Exception as exc:
                failures.append(str(exc))

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        raise SystemExit(1)

    copied = 0
    for render_path in sorted(glob(os.path.join(args.scene_dir, "render_results", "reconstruct_*", "traj1", "render.mp4"))):
        src = render_path.replace("traj1", "traj0").replace("render.mp4", "traj_caption.json")
        dst = render_path.replace("render.mp4", "traj_caption.json")
        if os.path.exists(src):
            shutil.copy(src, dst)
            copied += 1
    print(f"recaptioned {len(caption_videos)} videos; copied {copied} reconstruct traj1 captions")


if __name__ == "__main__":
    main()
