#!/usr/bin/env python3
# 用途：后处理工具；按不透明度过滤导出的 Gaussian PLY。
"""Low-memory opacity filter for binary Gaussian PLY files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PLY_TO_DTYPE = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _threshold_suffix(threshold: float) -> str:
    text = f"{threshold:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def _read_header(path: Path) -> tuple[list[str], int]:
    with path.open("rb") as f:
        data = f.read(8192)
    marker = b"end_header\n"
    index = data.find(marker)
    if index < 0:
        marker = b"end_header\r\n"
        index = data.find(marker)
    if index < 0:
        raise RuntimeError(f"{path} does not look like a PLY file with a complete header")
    header_bytes = data[: index + len(marker)]
    return header_bytes.decode("ascii").splitlines(keepends=True), len(header_bytes)


def _parse_vertex_layout(lines: list[str]) -> tuple[str, int, np.dtype]:
    fmt = None
    vertex_count = None
    vertex_props: list[tuple[str, str]] = []
    current_element = None
    endian = None

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
            if fmt == "binary_little_endian":
                endian = "<"
            elif fmt == "binary_big_endian":
                endian = ">"
            else:
                raise RuntimeError(f"unsupported PLY format: {fmt}")
        elif parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if parts[1] == "list":
                raise RuntimeError("list properties in vertex are not supported")
            if parts[1] not in PLY_TO_DTYPE:
                raise RuntimeError(f"unsupported PLY property type: {parts[1]}")
            vertex_props.append((parts[2], endian + PLY_TO_DTYPE[parts[1]]))

    if fmt is None or endian is None:
        raise RuntimeError("missing PLY format line")
    if vertex_count is None:
        raise RuntimeError("missing vertex element")
    if not vertex_props:
        raise RuntimeError("vertex element has no properties")
    if "opacity" not in [name for name, _ in vertex_props]:
        raise RuntimeError("vertex element has no opacity property")
    return fmt, vertex_count, np.dtype(vertex_props)


def _rewrite_vertex_count(lines: list[str], count: int) -> bytes:
    out = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            out.append(f"element vertex {count}{newline}")
        else:
            out.append(line)
    return "".join(out).encode("ascii")


def _opacity_score(opacity: np.ndarray, opacity_space: str) -> np.ndarray:
    if opacity_space == "raw":
        return opacity
    clipped = np.clip(opacity, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def filter_ply(
    input_path: Path,
    output_path: Path,
    threshold: float,
    chunk_size: int,
    opacity_space: str,
) -> dict[str, object]:
    lines, header_size = _read_header(input_path)
    _, vertex_count, vertex_dtype = _parse_vertex_layout(lines)
    vertices = np.memmap(
        input_path,
        dtype=vertex_dtype,
        mode="r",
        offset=header_size,
        shape=(vertex_count,),
    )

    kept_count = 0
    for start in range(0, vertex_count, chunk_size):
        opacity = vertices[start : start + chunk_size]["opacity"]
        kept_count += int(np.count_nonzero(_opacity_score(opacity, opacity_space) >= threshold))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_header = _rewrite_vertex_count(lines, kept_count)
    vertex_bytes = vertex_count * vertex_dtype.itemsize
    tail_offset = header_size + vertex_bytes

    with output_path.open("wb") as out:
        out.write(new_header)
        for start in range(0, vertex_count, chunk_size):
            chunk = vertices[start : start + chunk_size]
            kept = chunk[_opacity_score(chunk["opacity"], opacity_space) >= threshold]
            if len(kept):
                out.write(kept.tobytes())
        with input_path.open("rb") as src:
            src.seek(tail_offset)
            while True:
                data = src.read(8 * 1024 * 1024)
                if not data:
                    break
                out.write(data)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "threshold": threshold,
        "opacity_space": opacity_space,
        "source_vertices": vertex_count,
        "kept_vertices": kept_count,
        "removed_vertices": vertex_count - kept_count,
        "kept_ratio": kept_count / vertex_count if vertex_count else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a Gaussian PLY by opacity without loading it fully into RAM.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.93)
    parser.add_argument(
        "--opacity-space",
        choices=("alpha", "raw"),
        default="alpha",
        help="alpha means threshold sigmoid(raw opacity), matching gsplat opacity semantics.",
    )
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()

    if args.output is None:
        suffix = _threshold_suffix(args.threshold)
        args.output = args.input.with_name(f"{args.input.stem}_clean_opacity_{suffix}{args.input.suffix}")

    stats = filter_ply(args.input, args.output, args.threshold, args.chunk_size, args.opacity_space)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
