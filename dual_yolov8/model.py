from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def autopad(kernel_size: int, padding: int | None = None, dilation: int = 1) -> int:
    if padding is not None:
        return padding
    return dilation * (kernel_size - 1) // 2


class FusedConv(nn.Module):
    """YOLO Conv block after BN fusion: Conv2d(bias=True) + SiLU."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, channels: int, shortcut: bool = True) -> None:
        super().__init__()
        self.cv1 = FusedConv(channels, channels, 3, 1)
        self.cv2 = FusedConv(channels, channels, 3, 1)
        self.add = shortcut

    def forward(self, x: Tensor) -> Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True) -> None:
        super().__init__()
        self.c = c2 // 2
        self.cv1 = FusedConv(c1, 2 * self.c, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, shortcut) for _ in range(n))
        self.cv2 = FusedConv((2 + n) * self.c, c2, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(block(y[-1]) for block in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        c_ = c1 // 2
        self.cv1 = FusedConv(c1, c_, 1, 1)
        self.cv2 = FusedConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class DFL(nn.Module):
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False)
        self.conv.requires_grad_(False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        b, _, a = x.shape
        x = x.view(b, 4, self.reg_max, a).transpose(2, 1).softmax(1)
        return self.conv(x).view(b, 4, a)


class DetectHead(nn.Module):
    def __init__(self, nc: int = 1, hidden_cls: int = 80, ch: Iterable[int] = (64, 128, 256)) -> None:
        super().__init__()
        self.nc = nc
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        ch = list(ch)
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                FusedConv(c, 64, 3, 1),
                FusedConv(64, 64, 3, 1),
                nn.Conv2d(64, 4 * self.reg_max, 1, bias=True),
            )
            for c in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                FusedConv(c, hidden_cls, 3, 1),
                FusedConv(hidden_cls, hidden_cls, 3, 1),
                nn.Conv2d(hidden_cls, nc, 1, bias=True),
            )
            for c in ch
        )
        self.dfl = DFL(self.reg_max)

    def forward_raw(self, features: list[Tensor]) -> list[Tensor]:
        return [torch.cat((self.cv2[i](x), self.cv3[i](x)), 1) for i, x in enumerate(features)]

    def forward(self, features: list[Tensor], decode: bool = True) -> Tensor | list[Tensor]:
        raw = self.forward_raw(features)
        return self.decode(raw) if decode else raw

    def decode(self, raw: list[Tensor]) -> Tensor:
        shape = raw[0].shape
        b = shape[0]
        box, cls = torch.cat([xi.view(b, self.no, -1) for xi in raw], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        anchors, strides = make_anchors(raw, self.stride.to(raw[0].device, raw[0].dtype))
        dbox = dist2bbox(self.dfl(box), anchors.unsqueeze(0), xywh=True) * strides.view(1, 1, -1)
        return torch.cat((dbox, cls.sigmoid()), 1)


def make_anchors(raw: list[Tensor], strides: Tensor, grid_cell_offset: float = 0.5) -> tuple[Tensor, Tensor]:
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = raw[i].shape
        sx = torch.arange(w, device=raw[i].device, dtype=raw[i].dtype) + grid_cell_offset
        sy = torch.arange(h, device=raw[i].device, dtype=raw[i].dtype) + grid_cell_offset
        yy, xx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((xx, yy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, device=raw[i].device, dtype=raw[i].dtype))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: Tensor, anchor_points: Tensor, xywh: bool = True) -> Tensor:
    lt, rb = distance.chunk(2, 1)
    x1y1 = anchor_points.transpose(1, 2) - lt
    x2y2 = anchor_points.transpose(1, 2) + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), 1)
    return torch.cat((x1y1, x2y2), 1)


class YoloV8nSharedTrunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.ModuleList(
            [
                FusedConv(3, 16, 3, 2),
                FusedConv(16, 32, 3, 2),
                C2f(32, 32, 1, True),
                FusedConv(32, 64, 3, 2),
                C2f(64, 64, 2, True),
                FusedConv(64, 128, 3, 2),
                C2f(128, 128, 2, True),
                FusedConv(128, 256, 3, 2),
                C2f(256, 256, 1, True),
                SPPF(256, 256, 5),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Identity(),
                C2f(384, 128, 1, False),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Identity(),
                C2f(192, 64, 1, False),
                FusedConv(64, 64, 3, 2),
                nn.Identity(),
                C2f(192, 128, 1, False),
                FusedConv(128, 128, 3, 2),
                nn.Identity(),
                C2f(384, 256, 1, False),
            ]
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        y: list[Tensor | None] = [None] * 22
        y[0] = self.model[0](x)
        y[1] = self.model[1](y[0])
        y[2] = self.model[2](y[1])
        y[3] = self.model[3](y[2])
        y[4] = self.model[4](y[3])
        y[5] = self.model[5](y[4])
        y[6] = self.model[6](y[5])
        y[7] = self.model[7](y[6])
        y[8] = self.model[8](y[7])
        y[9] = self.model[9](y[8])
        y[10] = self.model[10](y[9])
        y[11] = torch.cat((y[10], y[6]), 1)
        y[12] = self.model[12](y[11])
        y[13] = self.model[13](y[12])
        y[14] = torch.cat((y[13], y[4]), 1)
        y[15] = self.model[15](y[14])
        y[16] = self.model[16](y[15])
        y[17] = torch.cat((y[16], y[12]), 1)
        y[18] = self.model[18](y[17])
        y[19] = self.model[19](y[18])
        y[20] = torch.cat((y[19], y[9]), 1)
        y[21] = self.model[21](y[20])
        return [y[15], y[18], y[21]]  # type: ignore[list-item]


class SingleYolov8n(nn.Module):
    def __init__(self, nc: int = 80) -> None:
        super().__init__()
        self.shared = YoloV8nSharedTrunk()
        self.head = DetectHead(nc=nc, hidden_cls=80)

    def forward(self, images: Tensor, decode: bool = True) -> Tensor | list[Tensor]:
        return self.head(self.shared(images), decode=decode)


class DualYolov8n(nn.Module):
    def __init__(self, nc: int = 1) -> None:
        super().__init__()
        self.shared = YoloV8nSharedTrunk()
        self.front_head = DetectHead(nc=nc, hidden_cls=80)
        self.top_head = DetectHead(nc=nc, hidden_cls=80)

    def forward(
        self,
        front_images: Tensor,
        top_images: Tensor,
        decode: bool = True,
    ) -> tuple[Tensor | list[Tensor], Tensor | list[Tensor]]:
        front_features = self.shared(front_images)
        top_features = self.shared(top_images)
        return self.front_head(front_features, decode=decode), self.top_head(top_features, decode=decode)

    def freeze_shared(self) -> None:
        self.shared.requires_grad_(False)

    def unfreeze_shared_from(self, first_layer: int = 15) -> None:
        for idx, layer in enumerate(self.shared.model):
            layer.requires_grad_(idx >= first_layer)


@dataclass
class LoadReport:
    copied: list[str]
    missing: list[str]
    reshaped: list[str]


def _load_onnx_initializers(onnx_path: str | Path) -> dict[str, Tensor]:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    return {init.name: torch.from_numpy(numpy_helper.to_array(init).copy()) for init in model.graph.initializer}


def _copy_param(param: Tensor, value: Tensor, name: str, report: LoadReport) -> None:
    if tuple(param.shape) != tuple(value.shape):
        raise ValueError(f"shape mismatch for {name}: model={tuple(param.shape)} onnx={tuple(value.shape)}")
    with torch.no_grad():
        param.copy_(value.to(dtype=param.dtype))
    report.copied.append(name)


def _copy_fused_conv(module: FusedConv, init: dict[str, Tensor], prefix: str, report: LoadReport) -> None:
    for suffix, param in (("weight", module.conv.weight), ("bias", module.conv.bias)):
        name = f"{prefix}.{suffix}"
        if name not in init:
            report.missing.append(name)
            continue
        _copy_param(param, init[name], name, report)


def _copy_plain_conv(module: nn.Conv2d, init: dict[str, Tensor], prefix: str, report: LoadReport) -> None:
    weight_name = f"{prefix}.weight"
    if weight_name not in init:
        report.missing.append(weight_name)
    else:
        source = init[weight_name]
        if module.weight.shape[0] == 1 and source.shape[0] == 80:
            source = source[:1]
            report.reshaped.append(weight_name)
        _copy_param(module.weight, source, weight_name, report)
    if module.bias is not None:
        bias_name = f"{prefix}.bias"
        if bias_name not in init:
            report.missing.append(bias_name)
        else:
            source = init[bias_name]
            if module.bias.shape[0] == 1 and source.shape[0] == 80:
                source = source[:1]
                report.reshaped.append(bias_name)
            _copy_param(module.bias, source, bias_name, report)


def _copy_c2f(module: C2f, init: dict[str, Tensor], prefix: str, report: LoadReport) -> None:
    _copy_fused_conv(module.cv1, init, f"{prefix}.cv1.conv", report)
    for i, block in enumerate(module.m):
        _copy_fused_conv(block.cv1, init, f"{prefix}.m.{i}.cv1.conv", report)
        _copy_fused_conv(block.cv2, init, f"{prefix}.m.{i}.cv2.conv", report)
    _copy_fused_conv(module.cv2, init, f"{prefix}.cv2.conv", report)


def _copy_sppf(module: SPPF, init: dict[str, Tensor], prefix: str, report: LoadReport) -> None:
    _copy_fused_conv(module.cv1, init, f"{prefix}.cv1.conv", report)
    _copy_fused_conv(module.cv2, init, f"{prefix}.cv2.conv", report)


def _copy_trunk(shared: YoloV8nSharedTrunk, init: dict[str, Tensor], report: LoadReport) -> None:
    for idx, module in enumerate(shared.model):
        prefix = f"model.{idx}"
        if isinstance(module, FusedConv):
            _copy_fused_conv(module, init, f"{prefix}.conv", report)
        elif isinstance(module, C2f):
            _copy_c2f(module, init, prefix, report)
        elif isinstance(module, SPPF):
            _copy_sppf(module, init, prefix, report)


def _copy_detect_head(head: DetectHead, init: dict[str, Tensor], report: LoadReport) -> None:
    for i in range(3):
        _copy_fused_conv(head.cv2[i][0], init, f"model.22.cv2.{i}.0.conv", report)
        _copy_fused_conv(head.cv2[i][1], init, f"model.22.cv2.{i}.1.conv", report)
        _copy_plain_conv(head.cv2[i][2], init, f"model.22.cv2.{i}.2", report)
        _copy_fused_conv(head.cv3[i][0], init, f"model.22.cv3.{i}.0.conv", report)
        _copy_fused_conv(head.cv3[i][1], init, f"model.22.cv3.{i}.1.conv", report)
        _copy_plain_conv(head.cv3[i][2], init, f"model.22.cv3.{i}.2", report)
    _copy_plain_conv(head.dfl.conv, init, "model.22.dfl.conv", report)


def load_fused_yolov8n_from_onnx(
    model: DualYolov8n | SingleYolov8n,
    onnx_path: str | Path,
    strict: bool = True,
) -> LoadReport:
    init = _load_onnx_initializers(onnx_path)
    report = LoadReport(copied=[], missing=[], reshaped=[])
    if isinstance(model, SingleYolov8n):
        _copy_trunk(model.shared, init, report)
        _copy_detect_head(model.head, init, report)
    elif isinstance(model, DualYolov8n):
        _copy_trunk(model.shared, init, report)
        _copy_detect_head(model.front_head, init, report)
        _copy_detect_head(model.top_head, init, report)
    else:
        raise TypeError(f"unsupported model type: {type(model)!r}")
    if strict and report.missing:
        raise KeyError(f"missing ONNX initializers: {report.missing[:20]}")
    return report
