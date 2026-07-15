n#!/usr/bin/env python3
# 用途：导出工具；从训练好的 GS 数据渲染更长的漫游视频。
import argparse
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import tqdm


WORLDGEN_ROOT = Path("/root/autodl-tmp/upstream/HY-World-2.0-gh/hyworld2/worldgen")
sys.path.insert(0, str(WORLDGEN_ROOT))

from gs.traj import generate_bspline_path, generate_ellipse_path_z, generate_spiral_path
from world_gs_trainer import Config, Runner


def _load_runner(args):
    cfg = Config(
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        disable_viewer=True,
        disable_video=False,
        antialiased=True,
        use_mask_gaussian=True,
        use_anchor_protection=False,
        use_align_protection=False,
        depth_loss=False,
        normal_loss=False,
        sky_depth_from_pcd=False,
        export_mesh=False,
        save_ply=False,
        convert_to_spz=False,
        sh_degree=args.sh_degree,
        near_plane=args.near,
        far_plane=args.far,
    )
    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    ckpt = torch.load(args.ckpt, map_location=runner.device, weights_only=False)
    for key in runner.splats.keys():
        if key in ckpt["splats"]:
            runner.splats[key].data = ckpt["splats"][key].to(runner.device)
    runner.is_training = False
    return runner, int(ckpt.get("step", 0))


def _resize_intrinsics(K, original_size, width, height):
    original_width, original_height = original_size
    K = K.copy()
    K[0, :] *= width / original_width
    K[1, :] *= height / original_height
    return K


def _resample_path(camtoworlds, frames):
    if len(camtoworlds) == frames:
        return camtoworlds
    idx = np.linspace(0, len(camtoworlds) - 1, frames, endpoint=True)
    lo = np.floor(idx).astype(np.int64)
    hi = np.ceil(idx).astype(np.int64)
    take_hi = (idx - lo) >= 0.5
    return camtoworlds[np.where(take_hi, hi, lo)]


def _drop_near_duplicate_poses(camtoworlds, min_step=1e-4):
    kept = [0]
    positions = camtoworlds[:, :3, 3]
    last = positions[0]
    for i in range(1, len(camtoworlds)):
        if np.linalg.norm(positions[i] - last) >= min_step:
            kept.append(i)
            last = positions[i]
    if len(kept) < 4:
        return camtoworlds
    return camtoworlds[np.array(kept, dtype=np.int64)]


def _make_paths(runner, frames):
    base = runner.parser.camtoworlds[5:-23]
    base = _drop_near_duplicate_poses(base)
    if len(base) > 96:
        base = base[np.linspace(0, len(base) - 1, 96, dtype=np.int64)]

    smooth = generate_bspline_path(base, n_points=frames, degree=3, smoothness=0.0)

    height = float(base[:, 2, 3].mean())
    ellipse = generate_ellipse_path_z(
        base,
        n_frames=frames,
        variation=0.15,
        phase=0.15,
        height=height,
    )

    spiral = generate_spiral_path(
        base,
        bounds=runner.parser.bounds * runner.scene_scale,
        n_frames=frames,
        n_rots=1.25,
        zrate=0.35,
        spiral_scale_r=float(runner.parser.extconf.get("spiral_radius_scale", 1.0)),
        focus_distance=0.72,
    )
    spiral = _resample_path(spiral, frames)

    return {
        "smooth_nav": smooth,
        "orbit_wide": ellipse,
        "spiral_focus": spiral,
    }


@torch.no_grad()
def _render_video(runner, name, camtoworlds, out_path, width, height, fps, crf):
    device = runner.device
    original_size = list(runner.parser.imsize_dict.values())[0]
    K_np = _resize_intrinsics(
        np.array(list(runner.parser.Ks_dict.values())[0], dtype=np.float32),
        original_size,
        width,
        height,
    )
    K = torch.from_numpy(K_np).float().to(device)

    c2w = np.concatenate(
        [
            camtoworlds,
            np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32), len(camtoworlds), axis=0),
        ],
        axis=1,
    )
    c2w = torch.from_numpy(c2w).float().to(device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        quality=10,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "slow"],
    )
    try:
        for i in tqdm.trange(len(c2w), desc=f"render {name}"):
            renders, _, _ = runner.rasterize_splats(
                camtoworlds=c2w[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=runner.cfg.sh_degree,
                near_plane=runner.cfg.near_plane,
                far_plane=runner.cfg.far_plane,
                render_mode="RGB",
            )
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)[0]
            frame = (colors.detach().cpu().numpy() * 255.0).astype(np.uint8)
            writer.append_data(frame)
            if i % 24 == 0:
                torch.cuda.empty_cache()
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/root/autodl-tmp/outputs/HY72/scene/gs_data")
    parser.add_argument("--result-dir", default="/root/autodl-tmp/outputs/HY72/gs_result")
    parser.add_argument("--ckpt", default="/root/autodl-tmp/outputs/HY72/gs_result/ckpts/ckpt_7999_rank0.pt")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/outputs/HY72/gs_result/long_renders")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--sh-degree", type=int, default=0)
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=1e10)
    parser.add_argument("--only", nargs="*", choices=["smooth_nav", "orbit_wide", "spiral_focus"])
    args = parser.parse_args()

    frames = args.fps * args.seconds
    runner, step = _load_runner(args)
    paths = _make_paths(runner, frames)
    if args.only:
        paths = {k: v for k, v in paths.items() if k in args.only}

    out_dir = Path(args.out_dir)
    for name, camtoworlds in paths.items():
        out_path = out_dir / f"{name}_{args.width}x{args.height}_{args.seconds}s_step{step}.mp4"
        _render_video(runner, name, camtoworlds, out_path, args.width, args.height, args.fps, args.crf)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
