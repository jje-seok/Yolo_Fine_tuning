from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

from .data import IMAGE_EXTS, letterbox


COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare 100 COCO-style letterboxed calibration images for DEEPX compile.")
    p.add_argument("--out", default="calibration/coco100")
    p.add_argument("--raw-root", default="datasets/raw")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--url", default=COCO128_URL)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def ensure_coco128(raw_root: Path, url: str) -> Path:
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root = raw_root / "coco128"
    if dataset_root.exists():
        return dataset_root

    zip_path = raw_root / "coco128.zip"
    if not zip_path.exists():
        print(f"downloading {url} -> {zip_path}")
        urllib.request.urlretrieve(url, zip_path)

    print(f"extracting {zip_path} -> {raw_root}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"expected extracted dataset at {dataset_root}")
    return dataset_root


def prepare_images(dataset_root: Path, out_dir: Path, count: int, imgsz: int, overwrite: bool) -> list[Path]:
    existing = sorted(p for p in out_dir.glob("*.jpg"))
    if len(existing) >= count and not overwrite:
        print(f"using existing calibration images: {out_dir}")
        return existing[:count]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_dir = dataset_root / "images" / "train2017"
    images = sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if len(images) < count:
        raise ValueError(f"need {count} source images, found {len(images)} under {source_dir}")

    written: list[Path] = []
    for idx, image_path in enumerate(images[:count]):
        with Image.open(image_path) as src:
            image, *_ = letterbox(src, imgsz)
        out_path = out_dir / f"coco_calib_{idx:04d}.jpg"
        image.save(out_path, quality=95)
        written.append(out_path)
    return written


def main() -> None:
    args = parse_args()
    dataset_root = ensure_coco128(Path(args.raw_root), args.url)
    written = prepare_images(dataset_root, Path(args.out), args.count, args.imgsz, args.overwrite)
    print(f"calibration images: {len(written)} -> {Path(args.out)}")


if __name__ == "__main__":
    main()
