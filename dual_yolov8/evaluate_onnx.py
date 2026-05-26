from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

from .data import IMAGE_EXTS, letterbox, read_yolo_label, remap_boxes_for_letterbox


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
    p = argparse.ArgumentParser(description="Compare original YOLOv8n ONNX and fine-tuned split ONNX on YOLO labels.")
    p.add_argument("--original-onnx", default="YOLOV8N.onnx")
    p.add_argument("--split-onnx-dir", default="runs/dual_yolov8_person/deepx_split_onnx")
    p.add_argument("--front-images", required=True)
    p.add_argument("--front-labels", required=True)
    p.add_argument("--top-images", required=True)
    p.add_argument("--top-labels", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--nms-iou", type=float, default=0.7)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--providers", nargs="+", default=None)
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


def preprocess_image(image_path: Path, imgsz: int) -> tuple[np.ndarray, tuple[float, int, int, int, int]]:
    image, scale, pad_x, pad_y, src_w, src_h = letterbox(Image.open(image_path), imgsz)
    arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return arr, (scale, pad_x, pad_y, src_w, src_h)


def load_ground_truth(label_path: Path, letterbox_info: tuple[float, int, int, int, int], imgsz: int) -> np.ndarray:
    scale, pad_x, pad_y, src_w, src_h = letterbox_info
    labels = read_yolo_label(label_path)
    labels = remap_boxes_for_letterbox(labels, scale, pad_x, pad_y, src_w, src_h, imgsz)
    if labels.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return xywh_to_xyxy(labels[:, 1:5] * imgsz).astype(np.float32)


def decode_person_output(output: np.ndarray, conf_thr: float, nms_iou: float, person_channel: int) -> tuple[np.ndarray, np.ndarray]:
    pred = output[0]
    boxes_xywh = pred[:4].T.astype(np.float32)
    scores = pred[person_channel].astype(np.float32)
    mask = scores >= conf_thr
    if not np.any(mask):
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    boxes = xywh_to_xyxy(boxes_xywh[mask])
    scores = scores[mask]
    keep = nms(boxes, scores, nms_iou)
    return boxes[keep], scores[keep]


class OriginalYoloRunner:
    def __init__(self, onnx_path: str | Path, providers: list[str]) -> None:
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: np.ndarray, conf_thr: float, nms_iou: float) -> tuple[np.ndarray, np.ndarray]:
        output = self.session.run(None, {self.input_name: image.astype(np.float32)})[0]
        return decode_person_output(output, conf_thr, nms_iou, person_channel=4)


class SplitYoloRunner:
    def __init__(self, split_dir: str | Path, head: str, providers: list[str]) -> None:
        split_dir = Path(split_dir)
        self.shared = ort.InferenceSession(str(split_dir / "shared_backbone.onnx"), providers=providers)
        self.head = ort.InferenceSession(str(split_dir / f"{head}_head.onnx"), providers=providers)

    def predict(self, image: np.ndarray, conf_thr: float, nms_iou: float) -> tuple[np.ndarray, np.ndarray]:
        p3, p4, p5 = self.shared.run(None, {"images": image.astype(np.float32)})
        output = self.head.run(None, {"p3": p3, "p4": p4, "p5": p5})[0]
        return decode_person_output(output, conf_thr, nms_iou, person_channel=4)


def image_files(images_dir: str | Path, max_images: int | None = None) -> list[Path]:
    images = sorted(p for p in Path(images_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return images[:max_images] if max_images else images


def collect_predictions(
    runner: OriginalYoloRunner | SplitYoloRunner,
    images_dir: str | Path,
    labels_dir: str | Path,
    imgsz: int,
    conf_thr: float,
    nms_iou: float,
    max_images: int | None,
) -> tuple[dict[int, np.ndarray], list[tuple[int, float, np.ndarray]]]:
    labels_dir = Path(labels_dir)
    gts: dict[int, np.ndarray] = {}
    preds: list[tuple[int, float, np.ndarray]] = []
    images = image_files(images_dir, max_images)
    if not images:
        raise FileNotFoundError(f"no images found under {images_dir}")

    root = Path(images_dir)
    for image_id, image_path in enumerate(tqdm(images, desc=f"eval {root.parent.name}/{root.name}")):
        image, lb_info = preprocess_image(image_path, imgsz)
        rel = image_path.relative_to(root).with_suffix(".txt")
        gt = load_ground_truth(labels_dir / rel, lb_info, imgsz)
        boxes, scores = runner.predict(image, conf_thr, nms_iou)
        gts[image_id] = gt
        preds.extend((image_id, float(score), box.astype(np.float32)) for box, score in zip(boxes, scores))
    return gts, preds


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


def evaluate(
    name: str,
    runner: OriginalYoloRunner | SplitYoloRunner,
    images_dir: str | Path,
    labels_dir: str | Path,
    imgsz: int,
    conf_thr: float,
    nms_iou: float,
    max_images: int | None,
) -> EvalResult:
    gts, preds = collect_predictions(runner, images_dir, labels_dir, imgsz, conf_thr, nms_iou, max_images)
    precision50, recall50, ap50 = ap_at_iou(gts, preds, 0.5)
    aps = [ap_at_iou(gts, preds, thr)[2] for thr in np.arange(0.5, 0.96, 0.05)]
    result = EvalResult(
        images=len(gts),
        gt=sum(len(v) for v in gts.values()),
        predictions=len(preds),
        precision50=precision50,
        recall50=recall50,
        ap50=ap50,
        map50_95=float(np.mean(aps)),
    )
    print_result(name, result)
    return result


def print_result(name: str, result: EvalResult) -> None:
    print(
        f"{name:28s} "
        f"images={result.images:5d} gt={result.gt:5d} pred={result.predictions:5d} "
        f"P50={result.precision50:.4f} R50={result.recall50:.4f} "
        f"AP50={result.ap50:.4f} mAP50-95={result.map50_95:.4f}"
    )


def main() -> None:
    args = parse_args()
    providers = args.providers
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in providers if p in ort.get_available_providers()]
    if not providers:
        providers = ["CPUExecutionProvider"]
    print("ONNXRuntime providers:", providers)

    original = OriginalYoloRunner(args.original_onnx, providers)
    split_front = SplitYoloRunner(args.split_onnx_dir, "front", providers)
    split_top = SplitYoloRunner(args.split_onnx_dir, "top", providers)

    print("\nFront-view evaluation")
    original_front = evaluate(
        "original YOLOv8n",
        original,
        args.front_images,
        args.front_labels,
        args.imgsz,
        args.conf,
        args.nms_iou,
        args.max_images,
    )
    fine_front = evaluate(
        "fine-tuned front_head",
        split_front,
        args.front_images,
        args.front_labels,
        args.imgsz,
        args.conf,
        args.nms_iou,
        args.max_images,
    )

    print("\nTop-view evaluation")
    original_top = evaluate(
        "original YOLOv8n",
        original,
        args.top_images,
        args.top_labels,
        args.imgsz,
        args.conf,
        args.nms_iou,
        args.max_images,
    )
    fine_top = evaluate(
        "fine-tuned top_head",
        split_top,
        args.top_images,
        args.top_labels,
        args.imgsz,
        args.conf,
        args.nms_iou,
        args.max_images,
    )

    print("\nAP50 delta")
    print(f"front fine-tuned - original: {fine_front.ap50 - original_front.ap50:+.4f}")
    print(f"top   fine-tuned - original: {fine_top.ap50 - original_top.ap50:+.4f}")


if __name__ == "__main__":
    main()
