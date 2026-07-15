#!/usr/bin/env python3
# 用途：检查工具；CUDA 阶段启动前确认存在可用 GPU，否则提前失败。
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Fail fast when a HY2 stage requires CUDA.")
    parser.add_argument("--stage", default="this stage")
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:
        print(f"[MISSING] {args.stage} requires CUDA, but torch import failed: {exc}", file=sys.stderr)
        raise SystemExit(2)

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        print(
            f"[MISSING] {args.stage} requires a CUDA GPU, but torch.cuda.is_available() is False "
            f"(device_count={torch.cuda.device_count()}).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    names = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
    print(f"[require_cuda] {args.stage}: {torch.cuda.device_count()} CUDA GPU(s) available: {names}")


if __name__ == "__main__":
    main()
