"""Dual-input shared YOLOv8n fine-tuning utilities."""

__all__ = ["DualYolov8n", "SingleYolov8n", "load_fused_yolov8n_from_onnx"]


def __getattr__(name):
    if name in __all__:
        from .model import DualYolov8n, SingleYolov8n, load_fused_yolov8n_from_onnx

        return {
            "DualYolov8n": DualYolov8n,
            "SingleYolov8n": SingleYolov8n,
            "load_fused_yolov8n_from_onnx": load_fused_yolov8n_from_onnx,
        }[name]
    raise AttributeError(name)
