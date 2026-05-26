from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox(image: Image.Image, size: int = 640, color: tuple[int, int, int] = (114, 114, 114)):
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), color)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y, w, h


def read_yolo_label(label_path: Path) -> np.ndarray:
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, x, y, w, h = map(float, parts[:5])
        if int(cls) == 0:
            rows.append([0.0, x, y, w, h])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def remap_boxes_for_letterbox(labels: np.ndarray, scale: float, pad_x: int, pad_y: int, src_w: int, src_h: int, size: int):
    if labels.size == 0:
        return labels
    out = labels.copy()
    x = labels[:, 1] * src_w
    y = labels[:, 2] * src_h
    w = labels[:, 3] * src_w
    h = labels[:, 4] * src_h
    out[:, 1] = (x * scale + pad_x) / size
    out[:, 2] = (y * scale + pad_y) / size
    out[:, 3] = (w * scale) / size
    out[:, 4] = (h * scale) / size
    out[:, 1:5] = np.clip(out[:, 1:5], 0.0, 1.0)
    return out


class YoloPersonDataset(Dataset):
    def __init__(self, images_dir: str | Path, labels_dir: str | Path, imgsz: int = 640, max_samples: int | None = None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.imgsz = imgsz
        self.images = sorted(p for p in self.images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        if max_samples:
            self.images = self.images[:max_samples]
        if not self.images:
            raise FileNotFoundError(f"no images found under {self.images_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[index]
        rel = img_path.relative_to(self.images_dir).with_suffix(".txt")
        label_path = self.labels_dir / rel
        image, scale, pad_x, pad_y, src_w, src_h = letterbox(Image.open(img_path), self.imgsz)
        labels = remap_boxes_for_letterbox(read_yolo_label(label_path), scale, pad_x, pad_y, src_w, src_h, self.imgsz)
        arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(arr), torch.from_numpy(labels)


class DualPersonDataset(Dataset):
    def __init__(self, front: YoloPersonDataset, top: YoloPersonDataset) -> None:
        self.front = front
        self.top = top

    def __len__(self) -> int:
        return max(len(self.front), len(self.top))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        front_img, front_labels = self.front[index % len(self.front)]
        top_img, top_labels = self.top[random.randrange(len(self.top))]
        return {
            "front_images": front_img,
            "top_images": top_img,
            "front_targets": front_labels,
            "top_targets": top_labels,
        }


def dual_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    return {
        "front_images": torch.stack([b["front_images"] for b in batch]),
        "top_images": torch.stack([b["top_images"] for b in batch]),
        "front_targets": [b["front_targets"] for b in batch],
        "top_targets": [b["top_targets"] for b in batch],
    }


def convert_coco_json_to_yolo_person(
    annotation_json: str | Path,
    image_root: str | Path,
    output_root: str | Path,
    split: str,
    max_images: int | None = None,
) -> None:
    annotation_json = Path(annotation_json)
    image_root = Path(image_root)
    output_root = Path(output_root)
    images_out = output_root / "images" / split
    labels_out = output_root / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    data = json.loads(annotation_json.read_text())
    categories = {c["id"]: c.get("name", "") for c in data.get("categories", [])}
    person_ids = {cid for cid, name in categories.items() if name.lower() == "person"}
    if not person_ids and categories:
        person_ids = {next(iter(categories))}

    images = data["images"][: max_images or None]
    allowed = {img["id"] for img in images}
    by_image: dict[int, list[list[float]]] = {img["id"]: [] for img in images}
    for ann in data.get("annotations", []):
        if ann.get("image_id") not in allowed or ann.get("category_id") not in person_ids or ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        img = next(i for i in images if i["id"] == ann["image_id"])
        cx = (x + w / 2) / img["width"]
        cy = (y + h / 2) / img["height"]
        by_image[img["id"]].append([0, cx, cy, w / img["width"], h / img["height"]])

    for img in images:
        src = image_root / img["file_name"]
        if src.exists():
            dst = images_out / Path(img["file_name"]).name
            if not dst.exists():
                shutil.copy2(src, dst)
        label_lines = [" ".join(f"{v:.6f}" if i else "0" for i, v in enumerate(row)) for row in by_image[img["id"]]]
        (labels_out / Path(img["file_name"]).with_suffix(".txt").name).write_text("\n".join(label_lines))


def filter_yolo_person_dataset(
    source_images: str | Path,
    source_labels: str | Path,
    output_root: str | Path,
    split: str,
    max_images: int | None = None,
) -> None:
    source_images = Path(source_images)
    source_labels = Path(source_labels)
    output_root = Path(output_root)
    images_out = output_root / "images" / split
    labels_out = output_root / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in source_images.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if max_images:
        images = images[:max_images]
    for img_path in images:
        rel = img_path.relative_to(source_images)
        labels = read_yolo_label(source_labels / rel.with_suffix(".txt"))
        dst_image = images_out / rel.name
        if not dst_image.exists():
            shutil.copy2(img_path, dst_image)
        lines = [" ".join(f"{v:.6f}" if i else "0" for i, v in enumerate(row)) for row in labels]
        (labels_out / rel.with_suffix(".txt").name).write_text("\n".join(lines))


def prepare_calibration_images(
    source_images: str | Path,
    output_dir: str | Path,
    count: int = 100,
    skip_first: int = 64,
    imgsz: int = 640,
    prefix: str = "calib",
) -> list[Path]:
    """Save letterboxed calibration JPEGs from images not used by fine-tuning.

    The default assumes the notebook fine-tuning command used the first 64 sorted
    images via `--max-*-samples 64`. Calibration images are therefore selected
    from the sorted image list after that range.
    """

    source_images = Path(source_images)
    output_dir = Path(output_dir)
    images = sorted(p for p in source_images.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    selected = images[skip_first : skip_first + count]
    if len(selected) < count:
        raise ValueError(
            f"Need {count} calibration images after skipping {skip_first}, but found {len(selected)}. "
            f"Source has {len(images)} images: {source_images}"
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for index, image_path in enumerate(selected):
        image, *_ = letterbox(Image.open(image_path), imgsz)
        dst = output_dir / f"{prefix}_{index:04d}.jpg"
        image.save(dst, quality=95)
        written.append(dst)
    return written
