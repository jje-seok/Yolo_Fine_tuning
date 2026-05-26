from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DualPersonDataset, YoloPersonDataset, dual_collate
from .loss import SimpleYoloV8PersonLoss
from .model import DualYolov8n, load_fused_yolov8n_from_onnx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune dual-input shared YOLOv8n person detector.")
    p.add_argument("--onnx", default="YOLOV8N.onnx")
    p.add_argument("--front-images", required=True)
    p.add_argument("--front-labels", required=True)
    p.add_argument("--top-images", required=True)
    p.add_argument("--top-labels", required=True)
    p.add_argument("--front-val-images", default=None)
    p.add_argument("--front-val-labels", default=None)
    p.add_argument("--top-val-images", default=None)
    p.add_argument("--top-val-labels", default=None)
    p.add_argument("--out", default="runs/dual_yolov8_person")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--finetune-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-front-samples", type=int, default=None)
    p.add_argument("--max-top-samples", type=int, default=None)
    p.add_argument("--max-front-val-samples", type=int, default=100)
    p.add_argument("--max-top-val-samples", type=int, default=100)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--val-conf", type=float, default=0.25)
    p.add_argument("--val-nms-iou", type=float, default=0.7)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_trainable(model: DualYolov8n, stage: str) -> None:
    if stage == "heads":
        model.freeze_shared()
        model.front_head.requires_grad_(True)
        model.top_head.requires_grad_(True)
    elif stage == "tail":
        model.freeze_shared()
        model.unfreeze_shared_from(15)
        model.front_head.requires_grad_(True)
        model.top_head.requires_grad_(True)
    else:
        raise ValueError(stage)


def single_collate(batch: list[tuple[Tensor, Tensor]]) -> dict[str, object]:
    return {
        "images": torch.stack([item[0] for item in batch]),
        "targets": [item[1] for item in batch],
    }


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


def ap_at_iou(
    gts: dict[int, np.ndarray],
    preds: list[tuple[int, float, np.ndarray]],
    iou_thr: float = 0.5,
) -> tuple[float, float, float]:
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
) -> dict[str, float]:
    model.eval()
    gts: dict[int, np.ndarray] = {}
    preds: list[tuple[int, float, np.ndarray]] = []
    image_id = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"val {view}", leave=False):
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

    precision, recall, ap50 = ap_at_iou(gts, preds, 0.5)
    return {
        "images": float(len(gts)),
        "gt": float(sum(len(v) for v in gts.values())),
        "predictions": float(len(preds)),
        "precision50": precision,
        "recall50": recall,
        "ap50": ap50,
    }


def evaluate_validation(
    model: DualYolov8n,
    val_loaders: dict[str, DataLoader],
    device: torch.device,
    imgsz: int,
    conf_thr: float,
    nms_iou: float,
) -> dict[str, dict[str, float] | float]:
    metrics: dict[str, dict[str, float] | float] = {}
    ap_values = []
    for view, loader in val_loaders.items():
        view_metrics = evaluate_view(model, loader, view, device, imgsz, conf_thr, nms_iou)
        metrics[view] = view_metrics
        ap_values.append(view_metrics["ap50"])
        print(
            f"val {view}: images={int(view_metrics['images'])} gt={int(view_metrics['gt'])} "
            f"pred={int(view_metrics['predictions'])} P50={view_metrics['precision50']:.4f} "
            f"R50={view_metrics['recall50']:.4f} AP50={view_metrics['ap50']:.4f}"
        )
    metrics["mean_ap50"] = float(np.mean(ap_values)) if ap_values else 0.0
    return metrics


def train_stage(
    model: DualYolov8n,
    loader: DataLoader,
    criterion: SimpleYoloV8PersonLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    out_dir: Path,
    stage_name: str,
    imgsz: int,
    val_loaders: dict[str, DataLoader],
    val_every: int,
    val_conf: float,
    val_nms_iou: float,
    start_epoch: int = 0,
) -> tuple[int, float, float]:
    best_loss = float("inf")
    best_val_ap50 = -1.0
    epoch_id = start_epoch
    for _ in range(epochs):
        epoch_id += 1
        model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"{stage_name} epoch {epoch_id}")
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
        avg_loss = running / max(len(loader), 1)
        val_metrics = None
        if val_loaders and val_every > 0 and epoch_id % val_every == 0:
            val_metrics = evaluate_validation(model, val_loaders, device, imgsz, val_conf, val_nms_iou)
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch_id,
            "loss": avg_loss,
            "stage": stage_name,
            "class_names": ["person"],
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, out_dir / "best.pt")
        if val_metrics and float(val_metrics["mean_ap50"]) > best_val_ap50:
            best_val_ap50 = float(val_metrics["mean_ap50"])
            torch.save(checkpoint, out_dir / "best_val.pt")
        val_suffix = f" mean_AP50={float(val_metrics['mean_ap50']):.4f}" if val_metrics else ""
        print(f"{stage_name} epoch {epoch_id}: loss={avg_loss:.6f}{val_suffix}")
    return epoch_id, best_loss, best_val_ap50


def maybe_val_loader(
    images: str | None,
    labels: str | None,
    imgsz: int,
    max_samples: int | None,
    batch: int,
    workers: int,
) -> DataLoader | None:
    if not images or not labels:
        return None
    dataset = YoloPersonDataset(images, labels, imgsz, max_samples)
    return DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=single_collate)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    front = YoloPersonDataset(args.front_images, args.front_labels, args.imgsz, args.max_front_samples)
    top = YoloPersonDataset(args.top_images, args.top_labels, args.imgsz, args.max_top_samples)
    dataset = DualPersonDataset(front, top)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=dual_collate)
    val_loaders = {}
    front_val_loader = maybe_val_loader(
        args.front_val_images,
        args.front_val_labels,
        args.imgsz,
        args.max_front_val_samples,
        args.batch,
        args.workers,
    )
    top_val_loader = maybe_val_loader(
        args.top_val_images,
        args.top_val_labels,
        args.imgsz,
        args.max_top_val_samples,
        args.batch,
        args.workers,
    )
    if front_val_loader:
        val_loaders["front"] = front_val_loader
    if top_val_loader:
        val_loaders["top"] = top_val_loader
    if val_loaders:
        print(f"validation enabled: {', '.join(val_loaders)}; every {args.val_every} epoch(s)")

    model = DualYolov8n(nc=1).to(device)
    report = load_fused_yolov8n_from_onnx(model, args.onnx, strict=True)
    print(f"loaded {len(report.copied)} tensors from {args.onnx}; person-cls tensors sliced={len(report.reshaped)}")

    criterion = SimpleYoloV8PersonLoss(imgsz=args.imgsz).to(device)
    epoch = 0
    if args.warmup_epochs:
        set_trainable(model, "heads")
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
        epoch, _, _ = train_stage(
            model,
            loader,
            criterion,
            optimizer,
            device,
            args.warmup_epochs,
            out_dir,
            "head-warmup",
            args.imgsz,
            val_loaders,
            args.val_every,
            args.val_conf,
            args.val_nms_iou,
            epoch,
        )
    if args.finetune_epochs:
        set_trainable(model, "tail")
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr * 0.5, weight_decay=1e-4)
        train_stage(
            model,
            loader,
            criterion,
            optimizer,
            device,
            args.finetune_epochs,
            out_dir,
            "tail-finetune",
            args.imgsz,
            val_loaders,
            args.val_every,
            args.val_conf,
            args.val_nms_iou,
            epoch,
        )


if __name__ == "__main__":
    main()
