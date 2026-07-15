#!/usr/bin/env python3
# 用途：后处理工具；对导出的 Gaussian PLY 做 viewer 坐标对齐。
"""Reorient a Gaussian PLY for browser viewers.

HY-World/GS exports carry their own learned up/facing directions in
position_meta_info.json. Many browser viewers assume a conventional
Y-up, -Z-forward world, so navigation can feel tilted even when the scene
looks visually correct. This script applies that rigid transform to both
Gaussian means and wxyz quaternions while streaming the PLY in chunks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PLY_TO_NUMPY = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def normalize(v: np.ndarray, name: str) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError(f"{name} has near-zero length: {v}")
    return v / n


def rotation_matrix_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a wxyz quaternion."""
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return normalize(q, "rotation quaternion")


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    out = np.empty_like(b)
    out[:, 0] = aw * bw - ax * bx - ay * by - az * bz
    out[:, 1] = aw * bx + ax * bw + ay * bz - az * by
    out[:, 2] = aw * by - ax * bz + ay * bw + az * bx
    out[:, 3] = aw * bz + ax * by - ay * bx + az * bw
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    out /= np.maximum(norm, 1e-8)
    return out


def build_alignment(meta: dict, target: str) -> tuple[np.ndarray, np.ndarray, dict]:
    up = normalize(np.asarray(meta["up_direction"], dtype=np.float64), "up_direction")
    facing = np.asarray(meta["facing_direction"], dtype=np.float64)
    facing = normalize(facing - up * np.dot(facing, up), "facing_direction projected")
    right = normalize(np.cross(facing, up), "right_direction")
    src_basis = np.stack([right, up, facing], axis=1)

    if target == "y-up":
        target_up = np.array([0.0, 1.0, 0.0])
        target_facing = np.array([0.0, 0.0, -1.0])
    elif target == "z-up":
        target_up = np.array([0.0, 0.0, 1.0])
        target_facing = np.array([0.0, 1.0, 0.0])
    else:
        raise ValueError(f"unknown target: {target}")

    target_right = normalize(np.cross(target_facing, target_up), "target_right")
    target_basis = np.stack([target_right, target_up, target_facing], axis=1)
    rotation = target_basis @ src_basis.T
    center = np.asarray(meta.get("center_point", [0.0, 0.0, 0.0]), dtype=np.float64)

    out_meta = dict(meta)
    out_meta["viewer_alignment"] = {
        "target": target,
        "source_up_direction": up.tolist(),
        "source_facing_direction": facing.tolist(),
        "target_up_direction": target_up.tolist(),
        "target_facing_direction": target_facing.tolist(),
        "rotation_matrix_source_to_viewer": rotation.tolist(),
        "source_center_point": center.tolist(),
        "note": "PLY positions are transformed as R @ (p - center); Gaussian quaternions are left-multiplied by R.",
    }
    out_meta["up_direction"] = target_up.tolist()
    out_meta["facing_direction"] = target_facing.tolist()
    out_meta["center_point"] = [0.0, 0.0, 0.0]
    return rotation.astype(np.float64), center, out_meta


def parse_ply_header(f) -> tuple[bytes, int, np.dtype, list[str]]:
    lines: list[bytes] = []
    vertex_count: int | None = None
    properties: list[tuple[str, str]] = []
    in_vertex = False

    while True:
        line = f.readline()
        if not line:
            raise ValueError("unexpected EOF while reading PLY header")
        lines.append(line)
        text = line.decode("ascii", errors="strict").strip()
        parts = text.split()
        if parts[:2] == ["format", "ascii"]:
            raise ValueError("ASCII PLY is not supported")
        if parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
        elif len(parts) >= 2 and parts[0] == "element" and parts[1] != "vertex":
            in_vertex = False
        elif in_vertex and len(parts) == 3 and parts[0] == "property":
            typ, name = parts[1], parts[2]
            if typ not in PLY_TO_NUMPY:
                raise ValueError(f"unsupported PLY property type: {typ}")
            properties.append((name, PLY_TO_NUMPY[typ]))
        elif text == "end_header":
            break

    if vertex_count is None:
        raise ValueError("PLY header has no vertex element")
    dtype = np.dtype(properties)
    return b"".join(lines), vertex_count, dtype, [name for name, _ in properties]


def transform_ply(input_path: Path, output_path: Path, meta_path: Path, target: str, chunk_size: int) -> Path:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rotation, center, out_meta = build_alignment(meta, target)
    q_align = rotation_matrix_to_quat_wxyz(rotation)

    with input_path.open("rb") as src:
        header, vertex_count, dtype, fields = parse_ply_header(src)
        required = {"x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3"}
        missing = sorted(required - set(fields))
        if missing:
            raise ValueError(f"PLY is missing required Gaussian fields: {missing}")

        if any(name.startswith("f_rest_") for name in fields):
            print("[align_viewer] warning: f_rest_* SH fields are preserved without SH rotation.", flush=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as dst:
            dst.write(header)
            remaining = vertex_count
            done = 0
            while remaining:
                n = min(chunk_size, remaining)
                arr = np.fromfile(src, dtype=dtype, count=n)
                if arr.shape[0] != n:
                    raise ValueError(f"expected {n} vertices, got {arr.shape[0]}")

                xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
                xyz = (xyz - center) @ rotation.T
                arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]

                quat = np.stack([arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"]], axis=1).astype(np.float64)
                quat = quat_mul_wxyz(q_align, quat)
                arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

                arr.tofile(dst)
                done += n
                remaining -= n
                if done == vertex_count or done % (chunk_size * 10) == 0:
                    print(f"[align_viewer] {done}/{vertex_count}", flush=True)

    out_meta_path = output_path.with_name(output_path.stem + "_position_meta_info.json")
    out_meta_path.write_text(json.dumps(out_meta, indent=2), encoding="utf-8")
    print(f"[align_viewer] wrote {output_path}", flush=True)
    print(f"[align_viewer] wrote {out_meta_path}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--target", choices=["y-up", "z-up"], default="y-up")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()

    input_path = args.input
    meta_path = args.meta or input_path.with_name("position_meta_info.json")
    output_path = args.output or input_path.with_name(input_path.stem + f"_viewer_{args.target.replace('-', '')}.ply")
    transform_ply(input_path, output_path, meta_path, args.target, args.chunk_size)


if __name__ == "__main__":
    main()
