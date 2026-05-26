from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create non-training calibration images for dual YOLOv8 quantization.")
    p.add_argument("--front-images", default="datasets/front_person/images/train")
    p.add_argument("--top-images", default="datasets/top_person/images/train")
    p.add_argument("--out", default="calibration")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--skip-first", type=int, default=64)
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from .data import prepare_calibration_images

    out = Path(args.out)
    front = prepare_calibration_images(
        source_images=args.front_images,
        output_dir=out / "front",
        count=args.count,
        skip_first=args.skip_first,
        imgsz=args.imgsz,
        prefix="front",
    )
    top = prepare_calibration_images(
        source_images=args.top_images,
        output_dir=out / "top",
        count=args.count,
        skip_first=args.skip_first,
        imgsz=args.imgsz,
        prefix="top",
    )
    print(f"front calibration images: {len(front)} -> {out / 'front'}")
    print(f"top calibration images: {len(top)} -> {out / 'top'}")


if __name__ == "__main__":
    main()
