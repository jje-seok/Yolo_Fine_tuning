from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch

from .model import DetectHead, DualYolov8n, YoloV8nSharedTrunk


class SharedBackboneExport(torch.nn.Module):
    def __init__(self, shared_backbone: YoloV8nSharedTrunk) -> None:
        super().__init__()
        self.shared_backbone = shared_backbone

    def forward(self, images: torch.Tensor):
        p3, p4, p5 = self.shared_backbone(images)
        return p3, p4, p5


class FrontHeadExport(torch.nn.Module):
    def __init__(self, front_head: DetectHead) -> None:
        super().__init__()
        self.front_head = front_head

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        return self.front_head([p3, p4, p5], decode=True)


class TopHeadExport(torch.nn.Module):
    def __init__(self, top_head: DetectHead) -> None:
        super().__init__()
        self.top_head = top_head

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        return self.top_head([p3, p4, p5], decode=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export DEEPX-friendly split YOLOv8n ONNX files.")
    p.add_argument("--checkpoint", default="runs/dual_yolov8_person/best.pt")
    p.add_argument("--out-dir", default="runs/dual_yolov8_person/deepx_split_onnx")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def load_checkpoint(checkpoint_path: str | Path) -> DualYolov8n:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = DualYolov8n(nc=1)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def _export(module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: Path, **kwargs) -> None:
    module.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "opset_version": kwargs.pop("opset_version"),
        "do_constant_folding": True,
        "dynamic_axes": None,
        **kwargs,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(module, args, str(output), **export_kwargs)
    print(f"exported {output}")


def export_split_onnx(model: DualYolov8n, out_dir: str | Path, imgsz: int = 640, opset: int = 17) -> dict[str, Path]:
    out_dir = Path(out_dir)
    dummy_image = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        p3, p4, p5 = model.shared(dummy_image)

    paths = {
        "shared_backbone": out_dir / "shared_backbone.onnx",
        "front_head": out_dir / "front_head.onnx",
        "top_head": out_dir / "top_head.onnx",
    }
    _export(
        SharedBackboneExport(model.shared),
        (dummy_image,),
        paths["shared_backbone"],
        input_names=["images"],
        output_names=["p3", "p4", "p5"],
        opset_version=opset,
    )
    _export(
        FrontHeadExport(model.front_head),
        (p3, p4, p5),
        paths["front_head"],
        input_names=["p3", "p4", "p5"],
        output_names=["front_output"],
        opset_version=opset,
    )
    _export(
        TopHeadExport(model.top_head),
        (p3, p4, p5),
        paths["top_head"],
        input_names=["p3", "p4", "p5"],
        output_names=["top_output"],
        opset_version=opset,
    )
    return paths


def main() -> None:
    args = parse_args()
    model = load_checkpoint(args.checkpoint)
    paths = export_split_onnx(model, args.out_dir, args.imgsz, args.opset)
    print("Expected files:")
    for path in paths.values():
        print(f"- {path}")
    print("Expected shapes:")
    print("- shared_backbone input: images [1, 3, 640, 640]")
    print("- shared_backbone outputs: p3 [1, 64, 80, 80], p4 [1, 128, 40, 40], p5 [1, 256, 20, 20]")
    print("- front_head output: front_output [1, 5, 8400]")
    print("- top_head output: top_output [1, 5, 8400]")


if __name__ == "__main__":
    main()
