from __future__ import annotations

import argparse
import csv
import random
import shutil
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from tqdm import tqdm

from .data import IMAGE_EXTS, letterbox


@dataclass(frozen=True)
class DatasetConfig:
    view: str
    head: str
    input_dir: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split and auto-label calibration images in YOLOv8 format.")
    p.add_argument(
        "--mode",
        choices=("create", "verify", "relabel"),
        default="create",
        help="create: build split dataset, verify: report label differences, relabel: overwrite only changed labels",
    )
    p.add_argument("--front-dir", default="calibration/front_calibration_dataset")
    p.add_argument("--top-dir", default="calibration/top_calibration_dataset")
    p.add_argument("--split-onnx-dir", default="runs/dual_yolov8_person/deepx_split_onnx")
    p.add_argument("--out-dir", default="calibration/labeled")
    p.add_argument("--views", nargs="+", choices=("front", "top"), default=["front", "top"])
    p.add_argument("--splits", nargs="+", choices=("train", "valid", "test"), default=["train", "valid", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms-iou", type=float, default=0.7)
    p.add_argument("--label-match-iou", type=float, default=0.95)
    p.add_argument("--report-name", default="label_validation_report.csv")
    p.add_argument("--preview-mismatches", action="store_true")
    p.add_argument("--preview-max", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260526)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--valid-ratio", type=float, default=0.20)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area1 = np.clip(box[2] - box[0], 0, None) * np.clip(box[3] - box[1], 0, None)
    area2 = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return inter / (area1 + area2 - inter + 1e-7)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break
        ious = box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_augmented_manifest(folder: Path) -> dict[str, str]:
    manifest = folder / "brightness_augmented_manifest.csv"
    if not manifest.exists():
        return {}
    mapping: dict[str, str] = {}
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["augmented_file"]] = row["source_file"]
    return mapping


def split_images(
    images: list[Path],
    augmented_to_source: dict[str, str],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
) -> dict[str, str]:
    groups: dict[str, list[Path]] = {}
    for image in images:
        group_key = augmented_to_source.get(image.name, image.name)
        groups.setdefault(group_key, []).append(image)

    group_keys = sorted(groups)
    random.Random(seed).shuffle(group_keys)
    train_count = int(len(group_keys) * train_ratio)
    valid_count = int(len(group_keys) * valid_ratio)
    split_by_group = {}
    for index, group_key in enumerate(group_keys):
        if index < train_count:
            split_by_group[group_key] = "train"
        elif index < train_count + valid_count:
            split_by_group[group_key] = "valid"
        else:
            split_by_group[group_key] = "test"

    split_by_image: dict[str, str] = {}
    for group_key, group_images in groups.items():
        for image in group_images:
            split_by_image[image.name] = split_by_group[group_key]
    return split_by_image


class SplitOnnxRunner:
    def __init__(self, split_onnx_dir: Path, head: str) -> None:
        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in ort.get_available_providers()]
        if not providers:
            providers = ["CPUExecutionProvider"]
        self.shared = ort.InferenceSession(str(split_onnx_dir / "shared_backbone.onnx"), providers=providers)
        self.head = ort.InferenceSession(str(split_onnx_dir / f"{head}_head.onnx"), providers=providers)

    def predict(self, image: np.ndarray) -> np.ndarray:
        p3, p4, p5 = self.shared.run(None, {"images": image.astype(np.float32)})
        return self.head.run(None, {"p3": p3, "p4": p4, "p5": p5})[0]


def preprocess(image_path: Path, imgsz: int) -> tuple[np.ndarray, tuple[float, int, int, int, int]]:
    with Image.open(image_path) as src:
        image, scale, pad_x, pad_y, src_w, src_h = letterbox(src, imgsz)
    arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return arr, (scale, pad_x, pad_y, src_w, src_h)


def labels_from_output(
    output: np.ndarray,
    letterbox_info: tuple[float, int, int, int, int],
    conf_thr: float,
    nms_iou: float,
) -> list[str]:
    scale, pad_x, pad_y, src_w, src_h = letterbox_info
    pred = output[0]
    boxes_xywh = pred[:4].T.astype(np.float32)
    scores = pred[4].astype(np.float32)
    mask = scores >= conf_thr
    if not np.any(mask):
        return []

    boxes = xywh_to_xyxy(boxes_xywh[mask])
    kept_scores = scores[mask]
    keep = nms(boxes, kept_scores, nms_iou)
    labels = []
    for box in boxes[keep]:
        x1 = np.clip((box[0] - pad_x) / scale, 0, src_w)
        y1 = np.clip((box[1] - pad_y) / scale, 0, src_h)
        x2 = np.clip((box[2] - pad_x) / scale, 0, src_w)
        y2 = np.clip((box[3] - pad_y) / scale, 0, src_h)
        w = x2 - x1
        h = y2 - y1
        if w <= 1 or h <= 1:
            continue
        cx = (x1 + x2) / 2 / src_w
        cy = (y1 + y2) / 2 / src_h
        nw = w / src_w
        nh = h / src_h
        values = np.clip([cx, cy, nw, nh], 0.0, 1.0)
        labels.append("0 " + " ".join(f"{v:.6f}" for v in values))
    return labels


def parse_label_lines(label_path: Path) -> tuple[np.ndarray, list[str]]:
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), ["missing_label"]

    boxes = []
    issues = []
    for line_index, raw_line in enumerate(label_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(f"line_{line_index}_bad_column_count")
            continue
        if parts[0] != "0":
            issues.append(f"line_{line_index}_bad_class_{parts[0]}")
        try:
            values = [float(v) for v in parts[1:]]
        except ValueError:
            issues.append(f"line_{line_index}_non_numeric")
            continue
        if values[2] <= 0 or values[3] <= 0:
            issues.append(f"line_{line_index}_non_positive_size")
        if any(v < 0.0 or v > 1.0 for v in values):
            issues.append(f"line_{line_index}_out_of_range")
        boxes.append(values)

    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), issues


def parse_generated_labels(labels: list[str]) -> np.ndarray:
    rows = []
    for label in labels:
        parts = label.split()
        if len(parts) == 5:
            rows.append([float(v) for v in parts[1:]])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def normalized_xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = np.empty_like(boxes, dtype=np.float32)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return np.clip(out, 0.0, 1.0)


def compare_label_boxes(existing_xywh: np.ndarray, predicted_xywh: np.ndarray, iou_thr: float) -> tuple[bool, float]:
    if len(existing_xywh) != len(predicted_xywh):
        return False, 0.0
    if len(existing_xywh) == 0:
        return True, 1.0

    existing = normalized_xywh_to_xyxy(existing_xywh)
    predicted = normalized_xywh_to_xyxy(predicted_xywh)
    ious = np.zeros((len(existing), len(predicted)), dtype=np.float32)
    for row_index, box in enumerate(existing):
        ious[row_index] = box_iou(box, predicted)

    matched_existing: set[int] = set()
    matched_predicted: set[int] = set()
    matched_ious = []
    while len(matched_existing) < len(existing):
        best_iou = -1.0
        best_pair = None
        for row_index in range(len(existing)):
            if row_index in matched_existing:
                continue
            for col_index in range(len(predicted)):
                if col_index in matched_predicted:
                    continue
                if ious[row_index, col_index] > best_iou:
                    best_iou = float(ious[row_index, col_index])
                    best_pair = (row_index, col_index)
        if best_pair is None:
            return False, 0.0
        matched_existing.add(best_pair[0])
        matched_predicted.add(best_pair[1])
        matched_ious.append(best_iou)

    min_iou = min(matched_ious) if matched_ious else 1.0
    return min_iou >= iou_thr, min_iou


def draw_boxes(
    image_path: Path,
    output_path: Path,
    existing_xywh: np.ndarray,
    predicted_xywh: np.ndarray,
) -> None:
    with Image.open(image_path) as src:
        image = src.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    def draw_set(boxes: np.ndarray, color: str, label: str) -> None:
        for box in boxes:
            cx, cy, bw, bh = box
            x1 = (cx - bw / 2) * width
            y1 = (cy - bh / 2) * height
            x2 = (cx + bw / 2) * width
            y2 = (cy + bh / 2) * height
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            draw.text((x1 + 3, y1 + 3), label, fill=color)

    draw_set(existing_xywh, "red", "current")
    draw_set(predicted_xywh, "lime", "pred")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_data_yaml(output_dir: Path) -> None:
    (output_dir / "data.yaml").write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/valid",
                "test: images/test",
                "names:",
                "  0: person",
                "",
            ]
        )
    )


def prepare_output_dirs(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    for split in ("train", "valid", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def process_dataset(config: DatasetConfig, split_onnx_dir: Path, args: argparse.Namespace) -> None:
    images = image_files(config.input_dir)
    if not images:
        raise FileNotFoundError(f"no images found under {config.input_dir}")

    augmented_to_source = load_augmented_manifest(config.input_dir)
    split_by_image = split_images(
        images=images,
        augmented_to_source=augmented_to_source,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )
    prepare_output_dirs(config.output_dir, args.overwrite)
    write_data_yaml(config.output_dir)

    runner = SplitOnnxRunner(split_onnx_dir, config.head)
    manifest_rows = []
    split_counts = {"train": 0, "valid": 0, "test": 0}
    label_counts = {"train": 0, "valid": 0, "test": 0}

    for image_path in tqdm(images, desc=f"autolabel {config.view}"):
        split = split_by_image[image_path.name]
        dst_image = config.output_dir / "images" / split / image_path.name
        dst_label = config.output_dir / "labels" / split / f"{image_path.stem}.txt"
        shutil.copy2(image_path, dst_image)

        image, lb_info = preprocess(image_path, args.imgsz)
        labels = labels_from_output(runner.predict(image), lb_info, args.conf, args.nms_iou)
        dst_label.write_text("\n".join(labels))

        source_image = augmented_to_source.get(image_path.name, image_path.name)
        is_augmented = image_path.name in augmented_to_source
        manifest_rows.append(
            {
                "image": image_path.name,
                "split": split,
                "source_image": source_image,
                "is_dark_augmented": str(is_augmented).lower(),
                "num_labels": str(len(labels)),
            }
        )
        split_counts[split] += 1
        label_counts[split] += len(labels)

    manifest_path = config.output_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "split", "source_image", "is_dark_augmented", "num_labels"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"{config.view}: images={len(images)} -> {config.output_dir}")
    for split in ("train", "valid", "test"):
        print(f"  {split}: images={split_counts[split]} labels={label_counts[split]}")


def refresh_manifest_counts(output_dir: Path) -> None:
    manifest_path = output_dir / "split_manifest.csv"
    if not manifest_path.exists():
        return

    rows = list(csv.DictReader(manifest_path.open(newline="")))
    for row in rows:
        label_path = output_dir / "labels" / row["split"] / Path(row["image"]).with_suffix(".txt").name
        boxes, _ = parse_label_lines(label_path)
        row["num_labels"] = str(len(boxes))

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "split", "source_image", "is_dark_augmented", "num_labels"])
        writer.writeheader()
        writer.writerows(rows)


def validate_labeled_dataset(
    config: DatasetConfig,
    split_onnx_dir: Path,
    args: argparse.Namespace,
    relabel: bool,
    backup_root: Path | None = None,
) -> None:
    output_dir = config.output_dir
    if not (output_dir / "images").exists():
        raise FileNotFoundError(f"{output_dir / 'images'} does not exist. Run --mode create first.")

    runner = SplitOnnxRunner(split_onnx_dir, config.head)
    report_rows = []
    total_images = 0
    different_count = 0
    invalid_count = 0
    relabeled_count = 0
    preview_count = 0

    for split in args.splits:
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        if not image_dir.exists():
            continue
        label_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

        for image_path in tqdm(images, desc=f"{'relabel' if relabel else 'verify'} {config.view}/{split}"):
            total_images += 1
            label_path = label_dir / f"{image_path.stem}.txt"
            existing_boxes, issues = parse_label_lines(label_path)
            image, lb_info = preprocess(image_path, args.imgsz)
            predicted_labels = labels_from_output(runner.predict(image), lb_info, args.conf, args.nms_iou)
            predicted_boxes = parse_generated_labels(predicted_labels)

            is_valid = not issues
            is_same, min_iou = compare_label_boxes(existing_boxes, predicted_boxes, args.label_match_iou) if is_valid else (False, 0.0)
            status = "same" if is_same else "different"
            if not is_valid:
                status = "invalid"
                invalid_count += 1
            elif not is_same:
                different_count += 1

            was_relabeled = False
            if relabel and not is_same:
                if backup_root is not None:
                    backup_label = backup_root / config.view / "labels" / split / label_path.name
                    backup_label.parent.mkdir(parents=True, exist_ok=True)
                    if label_path.exists():
                        shutil.copy2(label_path, backup_label)
                    else:
                        backup_label.with_suffix(".missing").write_text("")
                label_path.write_text("\n".join(predicted_labels))
                was_relabeled = True
                relabeled_count += 1

            if args.preview_mismatches and not is_same and preview_count < args.preview_max:
                preview_path = output_dir / "label_validation_previews" / split / image_path.name
                draw_boxes(image_path, preview_path, existing_boxes, predicted_boxes)
                preview_count += 1

            report_rows.append(
                {
                    "view": config.view,
                    "split": split,
                    "image": image_path.name,
                    "label": str(label_path.relative_to(output_dir)),
                    "status": status,
                    "issues": ";".join(issues),
                    "existing_labels": str(len(existing_boxes)),
                    "predicted_labels": str(len(predicted_boxes)),
                    "min_iou": f"{min_iou:.6f}",
                    "relabeled": str(was_relabeled).lower(),
                }
            )

    report_path = output_dir / args.report_name
    with report_path.open("w", newline="") as f:
        fieldnames = [
            "view",
            "split",
            "image",
            "label",
            "status",
            "issues",
            "existing_labels",
            "predicted_labels",
            "min_iou",
            "relabeled",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    if relabel:
        refresh_manifest_counts(output_dir)

    print(f"{config.view}: checked={total_images} invalid={invalid_count} different={different_count}")
    if relabel:
        print(f"{config.view}: relabeled={relabeled_count}")
    print(f"{config.view}: report={report_path}")
    if args.preview_mismatches:
        print(f"{config.view}: previews={output_dir / 'label_validation_previews'}")


def main() -> None:
    args = parse_args()
    split_onnx_dir = Path(args.split_onnx_dir)
    for filename in ("shared_backbone.onnx", "front_head.onnx", "top_head.onnx"):
        path = split_onnx_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)

    out_root = Path(args.out_dir)
    configs = [
        DatasetConfig("front", "front", Path(args.front_dir), out_root / "front"),
        DatasetConfig("top", "top", Path(args.top_dir), out_root / "top"),
    ]
    configs = [config for config in configs if config.view in args.views]

    if args.mode == "create":
        for config in configs:
            process_dataset(config, split_onnx_dir, args)
        return

    backup_root = None
    if args.mode == "relabel":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = out_root / "label_backups" / timestamp

    for config in configs:
        validate_labeled_dataset(config, split_onnx_dir, args, relabel=args.mode == "relabel", backup_root=backup_root)

    if backup_root is not None:
        print(f"backup={backup_root}")


if __name__ == "__main__":
    main()
