from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import IMAGE_EXTS, DualPersonDataset, dual_collate, letterbox, read_yolo_label, remap_boxes_for_letterbox
from .export import export_split_onnx
from .loss import SimpleYoloV8PersonLoss
from .model import DualYolov8n, load_fused_yolov8n_from_onnx
from .train import set_trainable


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    label_path: Path
    source: str
    split: str
    repeat: int = 0


@dataclass
class EvalResult:
    images: int
    gt: int
    predictions: int
    precision50: float
    recall50: float
    ap50: float
    map50_95: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Site fine-tune dual YOLOv8n with Roboflow + calibration data.")
    parser.add_argument("--onnx", default="YOLOV8N.onnx")
    parser.add_argument("--front-roboflow-root", required=True)
    parser.add_argument("--top-roboflow-root", required=True)
    parser.add_argument("--front-calib-root", default="calibration/labeled/front")
    parser.add_argument("--top-calib-root", default="calibration/labeled/top")
    parser.add_argument("--out", default="runs/dual_yolov8_site_mix")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--site-lr", type=float, default=2.5e-5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--mixed-finetune-epochs", type=int, default=6)
    parser.add_argument("--site-finetune-epochs", type=int, default=4)
    parser.add_argument("--calib-repeat", type=int, default=3)
    parser.add_argument("--roboflow-per-calib", type=int, default=2)
    parser.add_argument("--max-front-roboflow-samples", type=int, default=None)
    parser.add_argument("--max-top-roboflow-samples", type=int, default=None)
    parser.add_argument("--max-front-calib-samples", type=int, default=None)
    parser.add_argument("--max-top-calib-samples", type=int, default=None)
    parser.add_argument("--max-front-eval-samples", type=int, default=None)
    parser.add_argument("--max-top-eval-samples", type=int, default=None)
    parser.add_argument("--val-conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def yolo_split(root: str | Path, split: str) -> tuple[Path, Path]:
    root = Path(root)
    roboflow_images = root / split / "images"
    roboflow_labels = root / split / "labels"
    if roboflow_images.exists() and roboflow_labels.exists():
        return roboflow_images, roboflow_labels

    yolo_images = root / "images" / split
    yolo_labels = root / "labels" / split
    if yolo_images.exists() and yolo_labels.exists():
        return yolo_images, yolo_labels

    images = roboflow_images
    labels = roboflow_labels
    if not images.exists():
        raise FileNotFoundError(f"missing images for split '{split}': tried {roboflow_images} and {yolo_images}")
    if not labels.exists():
        raise FileNotFoundError(f"missing labels for split '{split}': tried {roboflow_labels} and {yolo_labels}")
    return images, labels


def collect_items(
    images_dir: Path,
    labels_dir: Path,
    source: str,
    split: str,
    max_items: int | None = None,
    seed: int = 20260526,
) -> list[DatasetItem]:
    images = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if max_items is not None and len(images) > max_items:
        images = sorted(random.Random(seed).sample(images, max_items))
    items = []
    for image_path in images:
        rel = image_path.relative_to(images_dir).with_suffix(".txt")
        items.append(DatasetItem(image_path, labels_dir / rel, source, split))
    if not items:
        raise FileNotFoundError(f"no images found under {images_dir}")
    return items


def sample_items(items: list[DatasetItem], limit: int, seed: int) -> list[DatasetItem]:
    if len(items) <= limit:
        return list(items)
    return sorted(random.Random(seed).sample(items, limit), key=lambda item: str(item.image_path))


def repeat_items(items: list[DatasetItem], repeats: int) -> list[DatasetItem]:
    repeated = []
    for repeat in range(repeats):
        repeated.extend(
            DatasetItem(item.image_path, item.label_path, item.source, item.split, repeat=repeat) for item in items
        )
    return repeated


class ImageListYoloPersonDataset(Dataset):
    def __init__(self, items: list[DatasetItem], imgsz: int = 640) -> None:
        self.items = items
        self.imgsz = imgsz
        if not self.items:
            raise ValueError("dataset is empty")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        item = self.items[index]
        image, scale, pad_x, pad_y, src_w, src_h = letterbox(Image.open(item.image_path), self.imgsz)
        labels = remap_boxes_for_letterbox(
            read_yolo_label(item.label_path),
            scale,
            pad_x,
            pad_y,
            src_w,
            src_h,
            self.imgsz,
        )
        arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(arr), torch.from_numpy(labels)


def single_collate(batch: list[tuple[Tensor, Tensor]]) -> dict[str, object]:
    return {
        "images": torch.stack([item[0] for item in batch]),
        "targets": [item[1] for item in batch],
    }


def write_manifest(path: Path, phase: str, front_items: list[DatasetItem], top_items: list[DatasetItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["phase", "view", "source", "split", "repeat", "image_path", "label_path"],
        )
        writer.writeheader()
        for view, items in (("front", front_items), ("top", top_items)):
            for item in items:
                writer.writerow(
                    {
                        "phase": phase,
                        "view": view,
                        "source": item.source,
                        "split": item.split,
                        "repeat": item.repeat,
                        "image_path": str(item.image_path),
                        "label_path": str(item.label_path),
                    }
                )


def assert_no_calibration_test_leak(
    front_train: list[DatasetItem],
    top_train: list[DatasetItem],
    front_calib_root: Path,
    top_calib_root: Path,
) -> None:
    leaked = []
    for view, root, items in (("front", front_calib_root, front_train), ("top", top_calib_root, top_train)):
        test_dir = root / "images" / "test"
        test_images = {p.resolve() for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
        for item in items:
            if item.source == "calibration" and item.image_path.resolve() in test_images:
                leaked.append(f"{view}:{item.image_path}")
    if leaked:
        raise RuntimeError("calibration test leakage detected:\n" + "\n".join(leaked[:20]))


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


def ap_at_iou(gts: dict[int, np.ndarray], preds: list[tuple[int, float, np.ndarray]], iou_thr: float) -> tuple[float, float, float]:
    total_gt = sum(len(v) for v in gts.values())
    if total_gt == 0:
        return 0.0, 0.0, 0.0

    matched = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gts.items()}
    sorted_preds = sorted(preds, key=lambda item: item[1], reverse=True)
    tp = np.zeros(len(sorted_preds), dtype=np.float32)
    fp = np.zeros(len(sorted_preds), dtype=np.float32)

    for idx, (image_id, _, pred_box) in enumerate(sorted_preds):
        gt_boxes = gts.get(image_id, np.zeros((0, 4), dtype=np.float32))
        if gt_boxes.size == 0:
            fp[idx] = 1.0
            continue
        ious = box_iou(pred_box, gt_boxes)
        best = int(ious.argmax())
        if ious[best] >= iou_thr and not matched[image_id][best]:
            tp[idx] = 1.0
            matched[image_id][best] = True
        else:
            fp[idx] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / (total_gt + 1e-12)
    precision = tp_cum / (tp_cum + fp_cum + 1e-12)
    ap = 0.0
    for recall_point in np.linspace(0, 1, 101):
        valid = precision[recall >= recall_point]
        ap += (valid.max() if valid.size else 0.0) / 101
    final_precision = float(precision[-1]) if precision.size else 0.0
    final_recall = float(recall[-1]) if recall.size else 0.0
    return final_precision, final_recall, float(ap)


def evaluate_view(
    model: DualYolov8n,
    loader: DataLoader,
    view: str,
    device: torch.device,
    imgsz: int,
    conf_thr: float,
    nms_iou: float,
) -> EvalResult:
    model.eval()
    gts: dict[int, np.ndarray] = {}
    preds: list[tuple[int, float, np.ndarray]] = []
    image_id = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval {view}", leave=False):
            images = batch["images"].to(device)  # type: ignore[union-attr]
            targets = batch["targets"]  # type: ignore[assignment]
            features = model.shared(images)
            output = model.front_head(features, decode=True) if view == "front" else model.top_head(features, decode=True)
            output_np = output.detach().cpu().numpy()

            for sample_idx, target in enumerate(targets):  # type: ignore[union-attr]
                labels = target.numpy()
                gt_boxes = (
                    xywh_to_xyxy(labels[:, 1:5] * imgsz).astype(np.float32)
                    if labels.size
                    else np.zeros((0, 4), dtype=np.float32)
                )
                pred = output_np[sample_idx]
                scores = pred[4].astype(np.float32)
                mask = scores >= conf_thr
                if np.any(mask):
                    boxes = xywh_to_xyxy(pred[:4].T.astype(np.float32)[mask])
                    kept_scores = scores[mask]
                    keep = nms(boxes, kept_scores, nms_iou)
                    preds.extend((image_id, float(kept_scores[i]), boxes[i].astype(np.float32)) for i in keep)
                gts[image_id] = gt_boxes
                image_id += 1

    precision50, recall50, ap50 = ap_at_iou(gts, preds, 0.5)
    aps = [ap_at_iou(gts, preds, float(thr))[2] for thr in np.arange(0.5, 0.96, 0.05)]
    return EvalResult(
        images=len(gts),
        gt=sum(len(v) for v in gts.values()),
        predictions=len(preds),
        precision50=precision50,
        recall50=recall50,
        ap50=ap50,
        map50_95=float(np.mean(aps)),
    )


def mean_result(front: EvalResult, top: EvalResult) -> EvalResult:
    return EvalResult(
        images=front.images + top.images,
        gt=front.gt + top.gt,
        predictions=front.predictions + top.predictions,
        precision50=(front.precision50 + top.precision50) / 2,
        recall50=(front.recall50 + top.recall50) / 2,
        ap50=(front.ap50 + top.ap50) / 2,
        map50_95=(front.map50_95 + top.map50_95) / 2,
    )


def evaluate_all(
    model: DualYolov8n,
    eval_loaders: dict[str, dict[str, DataLoader]],
    device: torch.device,
    imgsz: int,
    conf_thr: float,
    nms_iou: float,
    epoch: int,
    stage: str,
    loss: float | None,
) -> tuple[list[dict[str, str]], dict[str, dict[str, float]]]:
    rows = []
    summary: dict[str, dict[str, float]] = {}
    for split, loaders in eval_loaders.items():
        front_result = evaluate_view(model, loaders["front"], "front", device, imgsz, conf_thr, nms_iou)
        top_result = evaluate_view(model, loaders["top"], "top", device, imgsz, conf_thr, nms_iou)
        results = {"front": front_result, "top": top_result, "mean": mean_result(front_result, top_result)}
        summary[split] = {
            "mean_ap50": results["mean"].ap50,
            "mean_map50_95": results["mean"].map50_95,
        }
        for view, result in results.items():
            rows.append(
                {
                    "epoch": str(epoch),
                    "stage": stage,
                    "split": split,
                    "view": view,
                    "loss": "" if loss is None else f"{loss:.6f}",
                    "images": str(result.images),
                    "gt": str(result.gt),
                    "predictions": str(result.predictions),
                    "precision50": f"{result.precision50:.6f}",
                    "recall50": f"{result.recall50:.6f}",
                    "ap50": f"{result.ap50:.6f}",
                    "map50_95": f"{result.map50_95:.6f}",
                }
            )
        print(
            f"epoch {epoch} {stage} {split}: "
            f"front AP50={front_result.ap50:.4f} mAP={front_result.map50_95:.4f} | "
            f"top AP50={top_result.ap50:.4f} mAP={top_result.map50_95:.4f} | "
            f"mean AP50={results['mean'].ap50:.4f} mAP={results['mean'].map50_95:.4f}"
        )
    return rows, summary


def write_metrics(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "epoch",
        "stage",
        "split",
        "view",
        "loss",
        "images",
        "gt",
        "predictions",
        "precision50",
        "recall50",
        "ap50",
        "map50_95",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_one_epoch(
    model: DualYolov8n,
    loader: DataLoader,
    criterion: SimpleYoloV8PersonLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    stage: str,
) -> float:
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"{stage} epoch {epoch}")
    for batch in pbar:
        front_images = batch["front_images"].to(device)
        top_images = batch["top_images"].to(device)
        front_raw, top_raw = model(front_images, top_images, decode=False)
        front_loss, front_metrics = criterion(front_raw, batch["front_targets"])
        top_loss, top_metrics = criterion(top_raw, batch["top_targets"])
        loss = front_loss + top_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running += float(loss.detach().cpu())
        pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", pos=int(front_metrics["pos"] + top_metrics["pos"]))
    return running / max(len(loader), 1)


def save_checkpoint(
    path: Path,
    model: DualYolov8n,
    epoch: int,
    stage: str,
    loss: float | None,
    summary: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "epoch": epoch,
        "stage": stage,
        "loss": loss,
        "class_names": ["person"],
        "metrics": summary,
        "recipe": vars(args),
    }
    torch.save(checkpoint, path)


def load_checkpoint_into_model(path: Path) -> DualYolov8n:
    checkpoint = torch.load(path, map_location="cpu")
    model = DualYolov8n(nc=1)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def plot_metrics(history_csv: Path, output_png: Path) -> None:
    rows = list(csv.DictReader(history_csv.open(newline="")))
    rows = [row for row in rows if row["split"] == "test" and row["view"] in {"front", "top", "mean"}]
    if not rows:
        return

    epochs = sorted({int(row["epoch"]) for row in rows})
    width, height = 1200, 760
    margin_l, margin_r = 80, 40
    panel_h = 300
    top_a, top_b = 70, 420
    colors = {"front": (44, 123, 229), "top": (230, 126, 34), "mean": (40, 167, 69)}
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin_l, 24), "Calibration test metrics by epoch", fill=(20, 20, 20))

    def panel(metric: str, top: int, title: str) -> None:
        left, right = margin_l, width - margin_r
        bottom = top + panel_h
        draw.rectangle((left, top, right, bottom), outline=(190, 190, 190))
        draw.text((left, top - 24), title, fill=(20, 20, 20))
        for tick in range(0, 6):
            y = bottom - tick * panel_h / 5
            val = tick / 5
            draw.line((left, y, right, y), fill=(230, 230, 230))
            draw.text((20, y - 7), f"{val:.1f}", fill=(80, 80, 80))
        if len(epochs) == 1:
            x_for_epoch = {epochs[0]: (left + right) / 2}
        else:
            x_for_epoch = {
                epoch: left + (epoch - epochs[0]) * (right - left) / (epochs[-1] - epochs[0]) for epoch in epochs
            }
        for epoch in epochs:
            x = x_for_epoch[epoch]
            draw.line((x, bottom, x, bottom + 5), fill=(80, 80, 80))
            draw.text((x - 5, bottom + 8), str(epoch), fill=(80, 80, 80))
        for view, color in colors.items():
            points = []
            for row in rows:
                if row["view"] != view:
                    continue
                epoch = int(row["epoch"])
                value = max(0.0, min(1.0, float(row[metric])))
                x = x_for_epoch[epoch]
                y = bottom - value * panel_h
                points.append((x, y))
            points.sort()
            if len(points) == 1:
                x, y = points[0]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            elif len(points) > 1:
                draw.line(points, fill=color, width=3)
                for x, y in points:
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    panel("ap50", top_a, "AP50")
    panel("map50_95", top_b, "mAP50-95")
    legend_x = width - 300
    for idx, (view, color) in enumerate(colors.items()):
        y = 24 + idx * 22
        draw.line((legend_x, y + 7, legend_x + 28, y + 7), fill=color, width=3)
        draw.text((legend_x + 36, y), view, fill=(20, 20, 20))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)


def build_datasets(args: argparse.Namespace) -> tuple[dict[str, list[DatasetItem]], dict[str, dict[str, list[DatasetItem]]]]:
    front_calib_root = Path(args.front_calib_root)
    top_calib_root = Path(args.top_calib_root)

    front_calib_train = collect_items(*yolo_split(front_calib_root, "train"), "calibration", "train", args.max_front_calib_samples, args.seed)
    top_calib_train = collect_items(*yolo_split(top_calib_root, "train"), "calibration", "train", args.max_top_calib_samples, args.seed)
    front_robo_all = collect_items(*yolo_split(args.front_roboflow_root, "train"), "roboflow", "train", None, args.seed)
    top_robo_all = collect_items(*yolo_split(args.top_roboflow_root, "train"), "roboflow", "train", None, args.seed)

    front_robo_limit = len(front_calib_train) * args.roboflow_per_calib
    top_robo_limit = len(top_calib_train) * args.roboflow_per_calib
    if args.max_front_roboflow_samples is not None:
        front_robo_limit = min(front_robo_limit, args.max_front_roboflow_samples)
    if args.max_top_roboflow_samples is not None:
        top_robo_limit = min(top_robo_limit, args.max_top_roboflow_samples)

    front_robo = sample_items(front_robo_all, front_robo_limit, args.seed)
    top_robo = sample_items(top_robo_all, top_robo_limit, args.seed + 1)
    mixed_front = repeat_items(front_calib_train, args.calib_repeat) + front_robo
    mixed_top = repeat_items(top_calib_train, args.calib_repeat) + top_robo
    site_front = list(front_calib_train)
    site_top = list(top_calib_train)
    assert_no_calibration_test_leak(mixed_front + site_front, mixed_top + site_top, front_calib_root, top_calib_root)

    eval_items: dict[str, dict[str, list[DatasetItem]]] = {}
    for split in ("valid", "test"):
        eval_items[split] = {
            "front": collect_items(
                *yolo_split(front_calib_root, split),
                "calibration",
                split,
                args.max_front_eval_samples,
                args.seed,
            ),
            "top": collect_items(
                *yolo_split(top_calib_root, split),
                "calibration",
                split,
                args.max_top_eval_samples,
                args.seed,
            ),
        }

    train_items = {
        "mixed_front": mixed_front,
        "mixed_top": mixed_top,
        "site_front": site_front,
        "site_top": site_top,
    }
    return train_items, eval_items


def dual_loader(front_items: list[DatasetItem], top_items: list[DatasetItem], args: argparse.Namespace, shuffle: bool) -> DataLoader:
    dataset = DualPersonDataset(
        ImageListYoloPersonDataset(front_items, args.imgsz),
        ImageListYoloPersonDataset(top_items, args.imgsz),
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=shuffle,
        num_workers=args.workers,
        collate_fn=dual_collate,
        generator=generator,
    )


def eval_loaders(eval_items: dict[str, dict[str, list[DatasetItem]]], args: argparse.Namespace) -> dict[str, dict[str, DataLoader]]:
    loaders = {}
    for split, view_items in eval_items.items():
        loaders[split] = {
            "front": DataLoader(
                ImageListYoloPersonDataset(view_items["front"], args.imgsz),
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.workers,
                collate_fn=single_collate,
            ),
            "top": DataLoader(
                ImageListYoloPersonDataset(view_items["top"], args.imgsz),
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.workers,
                collate_fn=single_collate,
            ),
        }
    return loaders


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics_history.csv"
    plot_path = out_dir / "calibration_test_metrics.png"
    device = torch.device(args.device)

    train_items, eval_items = build_datasets(args)
    write_manifest(out_dir / "mixed_train_manifest.csv", "mixed", train_items["mixed_front"], train_items["mixed_top"])
    write_manifest(out_dir / "site_train_manifest.csv", "site", train_items["site_front"], train_items["site_top"])

    print(f"mixed front: {len(train_items['mixed_front'])} images/items")
    print(f"mixed top:   {len(train_items['mixed_top'])} images/items")
    print(f"site front:  {len(train_items['site_front'])} images")
    print(f"site top:    {len(train_items['site_top'])} images")

    mixed_loader = dual_loader(train_items["mixed_front"], train_items["mixed_top"], args, shuffle=True)
    site_loader = dual_loader(train_items["site_front"], train_items["site_top"], args, shuffle=True)
    loaders = eval_loaders(eval_items, args)

    model = DualYolov8n(nc=1).to(device)
    report = load_fused_yolov8n_from_onnx(model, args.onnx, strict=True)
    print(f"loaded {len(report.copied)} tensors from {args.onnx}; person-cls tensors sliced={len(report.reshaped)}")
    criterion = SimpleYoloV8PersonLoss(imgsz=args.imgsz).to(device)

    all_rows: list[dict[str, str]] = []
    rows, summary = evaluate_all(model, loaders, device, args.imgsz, args.val_conf, args.nms_iou, 0, "baseline", None)
    all_rows.extend(rows)
    write_metrics(metrics_path, all_rows)
    plot_metrics(metrics_path, plot_path)

    best_val = summary["valid"]["mean_map50_95"]
    best_test = summary["test"]["mean_map50_95"]
    save_checkpoint(out_dir / "best_val.pt", model, 0, "baseline", None, summary, args)
    save_checkpoint(out_dir / "best_test.pt", model, 0, "baseline", None, summary, args)

    epoch = 0
    stages = [
        ("mixed-head-warmup", "heads", mixed_loader, args.warmup_epochs, args.lr),
        ("mixed-tail-finetune", "tail", mixed_loader, args.mixed_finetune_epochs, args.lr * 0.5),
        ("site-tail-finetune", "tail", site_loader, args.site_finetune_epochs, args.site_lr),
    ]
    for stage_name, trainable, loader, epochs, lr in stages:
        if epochs <= 0:
            continue
        set_trainable(model, trainable)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            epoch += 1
            loss = train_one_epoch(model, loader, criterion, optimizer, device, epoch, stage_name)
            rows, summary = evaluate_all(
                model,
                loaders,
                device,
                args.imgsz,
                args.val_conf,
                args.nms_iou,
                epoch,
                stage_name,
                loss,
            )
            all_rows.extend(rows)
            write_metrics(metrics_path, all_rows)
            plot_metrics(metrics_path, plot_path)
            save_checkpoint(out_dir / "last.pt", model, epoch, stage_name, loss, summary, args)
            val_score = summary["valid"]["mean_map50_95"]
            test_score = summary["test"]["mean_map50_95"]
            if val_score > best_val:
                best_val = val_score
                save_checkpoint(out_dir / "best_val.pt", model, epoch, stage_name, loss, summary, args)
            if test_score > best_test:
                best_test = test_score
                save_checkpoint(out_dir / "best_test.pt", model, epoch, stage_name, loss, summary, args)
            print(
                f"{stage_name} epoch {epoch}: loss={loss:.6f} "
                f"valid_mean_mAP={val_score:.4f} test_mean_mAP={test_score:.4f}"
            )

    if not (out_dir / "last.pt").exists():
        save_checkpoint(out_dir / "last.pt", model, epoch, "baseline", None, summary, args)

    if not args.skip_export:
        best_model = load_checkpoint_into_model(out_dir / "best_val.pt")
        export_split_onnx(best_model, out_dir / "deepx_split_onnx", imgsz=args.imgsz, opset=args.opset)

    print(f"metrics: {metrics_path}")
    print(f"plot: {plot_path}")
    print(f"best_val: {out_dir / 'best_val.pt'}")
    print(f"best_test: {out_dir / 'best_test.pt'}")
    print(f"last: {out_dir / 'last.pt'}")
    if not args.skip_export:
        print(f"split ONNX: {out_dir / 'deepx_split_onnx'}")


if __name__ == "__main__":
    main()
