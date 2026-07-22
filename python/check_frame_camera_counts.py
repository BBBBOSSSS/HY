#!/usr/bin/env python3
# 用途：检查工具；验证 WorldStereo 视频帧数和 camera.json 是否一致。
import argparse
import json
import os
from glob import glob


def count_video_frames(path):
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        try:
            if cap.isOpened():
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if count > 0:
                    return count
        finally:
            cap.release()
    except Exception:
        pass

    try:
        import imageio.v3 as iio

        meta = iio.immeta(path)
        if meta.get("nframes"):
            return int(meta["nframes"])
    except Exception:
        pass

    try:
        from decord import VideoReader, cpu

        return len(VideoReader(path, ctx=cpu(0)))
    except Exception:
        return None


def allowed(camera_count, frame_count):
    return camera_count == frame_count or (camera_count == 81 and frame_count == 21)


def main():
    parser = argparse.ArgumentParser(description="Check HY-World generated-video frame counts against camera.json counts.")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--result-name", default="worldstereo-memory")
    args = parser.parse_args()

    patterns = [
        f"{args.scene_dir}/render_results/view*/*/{args.result_name}_result.mp4",
        f"{args.scene_dir}/render_results/target*/*/{args.result_name}_result.mp4",
        f"{args.scene_dir}/render_results/reconstruct*/traj0/{args.result_name}_result.mp4",
        f"{args.scene_dir}/render_results/wonder*/*/{args.result_name}_result.mp4",
    ]
    videos = sorted({p for pattern in patterns for p in glob(pattern)})
    if not videos:
        raise SystemExit(f"No result videos found under {args.scene_dir}/render_results")

    bad = []
    for video in videos:
        video_dir = os.path.dirname(video)
        camera_path = os.path.join(video_dir, "camera.json")
        if not os.path.exists(camera_path):
            bad.append((video, "missing camera.json", None, None))
            continue
        with open(camera_path) as f:
            cam = json.load(f)
        camera_count = len(cam["extrinsic"])
        frame_count = count_video_frames(video)
        if frame_count is None or not allowed(camera_count, frame_count):
            bad.append((video, "mismatch", camera_count, frame_count))

    print(f"checked {len(videos)} result videos")
    if bad:
        print("bad frame/camera counts:")
        for video, reason, camera_count, frame_count in bad[:80]:
            rel = os.path.relpath(video, args.scene_dir)
            print(f"  {rel}: {reason}, cameras={camera_count}, frames={frame_count}")
        raise SystemExit(1)
    print("all frame/camera counts are compatible")


if __name__ == "__main__":
    main()
