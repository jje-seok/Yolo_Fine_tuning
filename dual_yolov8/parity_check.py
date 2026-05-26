from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import torch

from .model import SingleYolov8n, load_fused_yolov8n_from_onnx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare original YOLOV8N.onnx output with PyTorch fused reconstruction.")
    p.add_argument("--onnx", default="YOLOV8N.onnx")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    image = torch.rand(1, 3, args.imgsz, args.imgsz)

    pt_model = SingleYolov8n(nc=80)
    load_fused_yolov8n_from_onnx(pt_model, args.onnx, strict=True)
    pt_model.eval()
    with torch.no_grad():
        pt_out = pt_model(image, decode=True).cpu().numpy()

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    ort_out = session.run(None, {"images": image.numpy().astype(np.float32)})[0]

    abs_diff = np.abs(pt_out - ort_out)
    print("pt shape:", pt_out.shape, "onnx shape:", ort_out.shape)
    print("max abs diff:", float(abs_diff.max()))
    print("mean abs diff:", float(abs_diff.mean()))
    if not np.allclose(pt_out, ort_out, atol=args.atol, rtol=args.rtol):
        raise SystemExit("parity check failed")
    print("parity check passed")


if __name__ == "__main__":
    main()
