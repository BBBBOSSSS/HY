#!/usr/bin/env python3
"""训练后固定分辨率导出：按训练分辨率×scale（默认 2）渲若干视角。

避免 viewer 无脑放大导致发糊。scale=1 为训练分辨率，scale=2 为 2×。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _setup_path() -> None:
    worldgen = Path(
        os.environ.get(
            "HY2_WORLDGEN_ROOT",
            "/root/autodl-tmp/_bundle_deps/upstream/HY-World-2.0-gh/hyworld2/worldgen",
        )
    )
    gsplat = Path(
        os.environ.get("HY2_GSPLAT_ROOT", str(worldgen / "third_party" / "gsplat_maskgaussian"))
    )
    for p in (str(worldgen), str(gsplat)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_cameras(cameras_json: Path) -> dict[str, dict]:
    raw = json.loads(cameras_json.read_text())
    out = {}
    for k, v in raw.items():
        if k in ("width", "height") or not isinstance(v, dict):
            continue
        if "extrinsic" in v and "intrinsic" in v:
            out[k] = v
    return out


def _pick_views(cam_ids: list[str], max_views: int) -> list[str]:
    if len(cam_ids) <= max_views:
        return cam_ids
    # prefer mid-trajectory reconstruct / view
    preferred = [c for c in cam_ids if any(p in c for p in ("reconstruct", "view", "target", "wonder"))]
    pool = preferred if preferred else cam_ids
    # group by traj
    from collections import defaultdict

    groups = defaultdict(list)
    for c in pool:
        groups[c.rsplit("_", 1)[0]].append(c)
    for g in groups:
        groups[g].sort()
    selected = []
    names = sorted(groups.keys())
    i = 0
    while len(selected) < max_views and names:
        g = names[i % len(names)]
        cand = groups[g]
        if cand:
            mid = cand[len(cand) // 2]
            if mid not in selected:
                selected.append(mid)
            groups[g] = [x for x in cand if x != mid]
        i += 1
        if i > max_views * len(names) + 10:
            break
    return selected[:max_views]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scale", type=float, default=2.0, help="1=train res, 2=2x")
    ap.add_argument("--sh-degree", type=int, default=2)
    ap.add_argument("--max-views", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for export_gs_fixed_res")

    _setup_path()
    from gsplat.rendering import rasterization

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cams = _load_cameras(data_dir / "cameras.json")
    images_dir = data_dir / "images"
    # train resolution from first existing image
    train_w = train_h = None
    for cid in cams:
        p = images_dir / f"{cid}.png"
        if p.exists():
            with Image.open(p) as im:
                train_w, train_h = im.size
            break
    if train_w is None:
        # fallback from K
        K0 = np.array(next(iter(cams.values()))["intrinsic"], dtype=np.float32).reshape(3, 3)
        train_w = int(round(float(K0[0, 2]) * 2))
        train_h = int(round(float(K0[1, 2]) * 2))

    scale = max(0.25, float(args.scale))
    width = int(round(train_w * scale))
    height = int(round(train_h * scale))
    # even dims for codecs / display
    width -= width % 2
    height -= height % 2

    device = torch.device(args.device)
    print(f"[export_fixed] train={train_w}x{train_h} scale={scale} -> {width}x{height}", flush=True)
    print(f"[export_fixed] loading {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    raw = ckpt["splats"]
    transform = np.asarray(ckpt.get("transform", np.eye(4)), dtype=np.float64)

    means = raw["means"].to(device)
    quats = F.normalize(raw["quats"].to(device), p=2, dim=-1)
    scales = torch.exp(raw["scales"].to(device))
    opacities = torch.sigmoid(raw["opacities"].to(device))
    sh0 = raw["sh0"].to(device)
    shN = raw["shN"].to(device)
    colors = torch.cat([sh0, shN], dim=-2)
    k_bands = colors.shape[-2]
    max_deg = int(math.sqrt(k_bands) - 1) if k_bands > 0 else 0
    sh_degree = min(int(args.sh_degree), max_deg)
    print(f"[export_fixed] N={means.shape[0]} sh_bands={k_bands} use_sh_degree={sh_degree}", flush=True)

    view_ids = _pick_views(sorted(cams.keys()), args.max_views)
    meta = {
        "train_resolution": [train_w, train_h],
        "export_resolution": [width, height],
        "scale": scale,
        "sh_degree": sh_degree,
        "views": [],
    }

    @torch.no_grad()
    def render_one(w2c: np.ndarray, K: np.ndarray) -> Image.Image:
        c2w = np.linalg.inv(np.asarray(w2c, dtype=np.float64).reshape(4, 4))
        c2w_n = (transform @ c2w).astype(np.float32)
        viewmat = torch.linalg.inv(torch.from_numpy(c2w_n).to(device))[None]
        # scale intrinsics to export size
        K = np.asarray(K, dtype=np.float32).reshape(3, 3).copy()
        sx = width / train_w
        sy = height / train_h
        K[0, :] *= sx
        K[1, :] *= sy
        K_t = torch.from_numpy(K).to(device)[None]
        render_colors, _, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmat,
            K_t,
            width,
            height,
            sh_degree=sh_degree,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            backgrounds=None,
        )
        rgb = torch.clamp(render_colors[0, ..., :3], 0, 1)
        arr = (rgb.float().cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    for i, cid in enumerate(view_ids):
        cam = cams[cid]
        img = render_one(cam["extrinsic"], cam["intrinsic"])
        # also save 1x if scale!=1 for comparison
        out_path = out_dir / f"{cid}_x{scale:g}_{width}x{height}.png"
        img.save(out_path)
        meta["views"].append({"camera_id": cid, "path": str(out_path)})
        print(f"[export_fixed] {i+1}/{len(view_ids)} {out_path.name}", flush=True)
        torch.cuda.empty_cache()

    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"[export_fixed] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
