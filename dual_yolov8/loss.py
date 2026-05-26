from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def xywhn_to_xyxy_pixels(boxes: Tensor, imgsz: int) -> Tensor:
    xy = boxes[:, :2] * imgsz
    wh = boxes[:, 2:] * imgsz
    return torch.cat((xy - wh / 2, xy + wh / 2), 1)


def xyxy_to_xywh(boxes: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), 1)


def bbox_iou_xyxy(box1: Tensor, box2: Tensor, eps: float = 1e-7) -> Tensor:
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)
    return inter / (area1 + area2 - inter + eps)


class SimpleYoloV8PersonLoss(nn.Module):
    """Small self-contained loss for Colab fine-tuning of the dual heads.

    It assigns each person box to the center cell on every YOLO scale, applies BCE
    to the person logit, and optimizes decoded boxes with L1 + IoU losses. This is
    intentionally lighter than the official Ultralytics TAL loss, but keeps the
    notebook independent from private Ultralytics internals.
    """

    def __init__(self, imgsz: int = 640, strides: tuple[int, int, int] = (8, 16, 32)) -> None:
        super().__init__()
        self.imgsz = imgsz
        self.strides = strides
        self.cls_weight = 1.0
        self.box_weight = 7.5
        self.iou_weight = 2.5

    def forward(self, raw_outputs: list[Tensor], targets: list[Tensor]) -> tuple[Tensor, dict[str, float]]:
        device = raw_outputs[0].device
        total_cls = torch.zeros((), device=device)
        total_box = torch.zeros((), device=device)
        total_iou = torch.zeros((), device=device)
        total_pos = 0

        for scale_idx, raw in enumerate(raw_outputs):
            b, c, h, w = raw.shape
            stride = self.strides[scale_idx]
            cls_logits = raw[:, 64:65]
            cls_target = torch.zeros_like(cls_logits)
            pos_pred_boxes = []
            pos_target_boxes = []

            decoded_xywh = self._decode_scale(raw, stride)
            decoded_xyxy = self._xywh_to_xyxy(decoded_xywh)

            for batch_idx, boxes in enumerate(targets):
                if boxes.numel() == 0:
                    continue
                boxes = boxes.to(device)
                gt_xyxy = xywhn_to_xyxy_pixels(boxes[:, 1:5], self.imgsz)
                centers = boxes[:, 1:3] * self.imgsz / stride
                gx = centers[:, 0].floor().long().clamp(0, w - 1)
                gy = centers[:, 1].floor().long().clamp(0, h - 1)
                cls_target[batch_idx, 0, gy, gx] = 1.0
                pos_pred_boxes.append(decoded_xyxy[batch_idx, gy, gx])
                pos_target_boxes.append(gt_xyxy)

            total_cls = total_cls + F.binary_cross_entropy_with_logits(cls_logits, cls_target, reduction="mean")
            if pos_pred_boxes:
                pred = torch.cat(pos_pred_boxes, 0)
                target = torch.cat(pos_target_boxes, 0)
                total_box = total_box + F.l1_loss(pred / self.imgsz, target / self.imgsz, reduction="mean")
                total_iou = total_iou + (1.0 - bbox_iou_xyxy(pred, target)).mean()
                total_pos += pred.shape[0]

        loss = self.cls_weight * total_cls + self.box_weight * total_box + self.iou_weight * total_iou
        metrics = {
            "loss": float(loss.detach().cpu()),
            "cls": float(total_cls.detach().cpu()),
            "box": float(total_box.detach().cpu()),
            "iou": float(total_iou.detach().cpu()),
            "pos": float(total_pos),
        }
        return loss, metrics

    def _decode_scale(self, raw: Tensor, stride: int) -> Tensor:
        b, _, h, w = raw.shape
        box_logits = raw[:, :64].view(b, 4, 16, h, w).softmax(2)
        proj = torch.arange(16, device=raw.device, dtype=raw.dtype).view(1, 1, 16, 1, 1)
        dist = (box_logits * proj).sum(2)
        yy, xx = torch.meshgrid(
            torch.arange(h, device=raw.device, dtype=raw.dtype),
            torch.arange(w, device=raw.device, dtype=raw.dtype),
            indexing="ij",
        )
        anchor_x = xx + 0.5
        anchor_y = yy + 0.5
        left, top, right, bottom = dist.unbind(1)
        x1 = (anchor_x - left) * stride
        y1 = (anchor_y - top) * stride
        x2 = (anchor_x + right) * stride
        y2 = (anchor_y + bottom) * stride
        return xyxy_to_xywh(torch.stack((x1, y1, x2, y2), -1).view(-1, 4)).view(b, h, w, 4)

    @staticmethod
    def _xywh_to_xyxy(boxes: Tensor) -> Tensor:
        xy = boxes[..., :2]
        wh = boxes[..., 2:]
        return torch.cat((xy - wh / 2, xy + wh / 2), -1)

