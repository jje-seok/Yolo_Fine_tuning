from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .export import FrontHeadExport, TopHeadExport, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify DEEPX split ONNX files against PyTorch.")
    p.add_argument("--checkpoint", default="runs/dual_yolov8_person/best.pt")
    p.add_argument("--onnx-dir", default="runs/dual_yolov8_person/deepx_split_onnx")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def main() -> None:
    args = parse_args()
    onnx_dir = Path(args.onnx_dir)
    shared_path = onnx_dir / "shared_backbone.onnx"
    front_path = onnx_dir / "front_head.onnx"
    top_path = onnx_dir / "top_head.onnx"
    for path in (shared_path, front_path, top_path):
        if not path.exists():
            raise FileNotFoundError(path)

    model = load_checkpoint(args.checkpoint)
    model.eval()
    image = torch.zeros(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        p3, p4, p5 = model.shared(image)
        pt_front = FrontHeadExport(model.front_head)(p3, p4, p5).cpu().numpy()
        pt_top = TopHeadExport(model.top_head)(p3, p4, p5).cpu().numpy()

    shared_sess = ort.InferenceSession(str(shared_path), providers=["CPUExecutionProvider"])
    front_sess = ort.InferenceSession(str(front_path), providers=["CPUExecutionProvider"])
    top_sess = ort.InferenceSession(str(top_path), providers=["CPUExecutionProvider"])

    ort_p3, ort_p4, ort_p5 = shared_sess.run(None, {"images": image.numpy().astype(np.float32)})
    ort_front = front_sess.run(None, {"p3": ort_p3, "p4": ort_p4, "p5": ort_p5})[0]
    ort_top = top_sess.run(None, {"p3": ort_p3, "p4": ort_p4, "p5": ort_p5})[0]

    print("PyTorch shapes:")
    print("- p3, p4, p5:", tuple(p3.shape), tuple(p4.shape), tuple(p5.shape))
    print("- front_output:", pt_front.shape)
    print("- top_output:", pt_top.shape)
    print("ONNXRuntime shapes:")
    print("- p3, p4, p5:", ort_p3.shape, ort_p4.shape, ort_p5.shape)
    print("- front_output:", ort_front.shape)
    print("- top_output:", ort_top.shape)
    print("Max absolute error:")
    print("- p3:", _max_abs(p3.cpu().numpy(), ort_p3))
    print("- p4:", _max_abs(p4.cpu().numpy(), ort_p4))
    print("- p5:", _max_abs(p5.cpu().numpy(), ort_p5))
    print("- front_output:", _max_abs(pt_front, ort_front))
    print("- top_output:", _max_abs(pt_top, ort_top))

    expected = (1, 5, 8400)
    if tuple(pt_front.shape) != expected or tuple(pt_top.shape) != expected:
        raise SystemExit(f"unexpected PyTorch output shapes: {pt_front.shape}, {pt_top.shape}")
    if tuple(ort_front.shape) != expected or tuple(ort_top.shape) != expected:
        raise SystemExit(f"unexpected ONNX output shapes: {ort_front.shape}, {ort_top.shape}")

    print("DEEPX split ONNX verification passed")


if __name__ == "__main__":
    main()
