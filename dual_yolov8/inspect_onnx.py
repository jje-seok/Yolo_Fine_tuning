from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def _shape(value_info):
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.dim_value else dim.dim_param)
    return dims


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect YOLOV8N.onnx graph boundaries and initializer names.")
    p.add_argument("onnx", nargs="?", default="YOLOV8N.onnx")
    args = p.parse_args()
    path = Path(args.onnx)
    model = onnx.load(str(path))
    print(f"path: {path}")
    print(f"producer: {model.producer_name} {model.producer_version}")
    print(f"opset: {[(o.domain, o.version) for o in model.opset_import]}")
    print("inputs:")
    for item in model.graph.input:
        print(f"  {item.name}: {_shape(item)}")
    print("outputs:")
    for item in model.graph.output:
        print(f"  {item.name}: {_shape(item)}")
    names = {i.name for i in model.graph.initializer}
    required = [
        "model.0.conv.weight",
        "model.21.cv2.conv.bias",
        "model.22.cv2.0.2.weight",
        "model.22.cv3.0.2.weight",
        "model.22.dfl.conv.weight",
    ]
    missing = [name for name in required if name not in names]
    if missing:
        raise SystemExit(f"missing expected YOLOv8n initializers: {missing}")
    print(f"initializers: {len(names)}; expected YOLOv8n names are present")


if __name__ == "__main__":
    main()

