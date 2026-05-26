from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data import IMAGE_EXTS, letterbox


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile split YOLOv8 ONNX files with DEEPX DX-COM.")
    p.add_argument("--split-onnx-dir", default="runs/dual_yolov8_person/deepx_split_onnx")
    p.add_argument("--calibration-images", default="calibration/coco100")
    p.add_argument("--out-dir", default="runs/dual_yolov8_person/deepx_compiled")
    p.add_argument("--calibration-num", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--calibration-method", choices=["ema", "minmax"], default="ema")
    p.add_argument("--quantization-device", default=None, help='Optional DX-COM device, e.g. "cpu", "cuda", "cuda:0".')
    p.add_argument("--opt-level", type=int, choices=[0, 1], default=1)
    p.add_argument("--aggressive-partitioning", action="store_true")
    p.add_argument("--gen-log", action="store_true")
    p.add_argument("--feature-cache-dir", default=None)
    p.add_argument("--skip-shared", action="store_true", help="Skip shared_backbone compilation and compile heads only.")
    p.add_argument("--skip-heads", action="store_true", help="Skip front/top head compilation.")
    return p.parse_args()


def calibration_image_files(image_dir: str | Path, count: int) -> list[Path]:
    images = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if len(images) < count:
        raise ValueError(f"need {count} calibration images, found {len(images)} under {image_dir}")
    return images[:count]


def preprocess_image(image_path: Path, imgsz: int) -> torch.Tensor:
    with Image.open(image_path) as src:
        image, *_ = letterbox(src, imgsz)
    arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr)


class ImageCalibrationDataset(Dataset):
    def __init__(self, image_dir: str | Path, count: int, imgsz: int) -> None:
        self.images = calibration_image_files(image_dir, count)
        self.imgsz = imgsz

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return preprocess_image(self.images[index], self.imgsz)


class FeatureCalibrationDataset(Dataset):
    def __init__(
        self,
        shared_backbone_onnx: str | Path,
        image_dir: str | Path,
        count: int,
        imgsz: int,
        cache_dir: str | Path | None = None,
    ) -> None:
        import onnxruntime as ort

        self.images = calibration_image_files(image_dir, count)
        self.imgsz = imgsz
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in ort.get_available_providers()]
        if not providers:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(shared_backbone_onnx), providers=providers)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_path = self.cache_dir / f"features_{index:04d}.npz" if self.cache_dir else None
        if cache_path and cache_path.exists():
            with np.load(cache_path) as data:
                features = tuple(data[name].astype(np.float32) for name in ("p3", "p4", "p5"))
            return tuple(torch.from_numpy(feature) for feature in features)  # type: ignore[return-value]

        image = preprocess_image(self.images[index], self.imgsz).numpy()[None].astype(np.float32)
        p3, p4, p5 = self.session.run(None, {"images": image})
        features = (p3[0].astype(np.float32), p4[0].astype(np.float32), p5[0].astype(np.float32))
        if cache_path:
            np.savez_compressed(cache_path, p3=features[0], p4=features[1], p5=features[2])
        return tuple(torch.from_numpy(feature) for feature in features)  # type: ignore[return-value]


def compile_model(dx_com, model_path: Path, output_dir: Path, dataloader: DataLoader, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "model": str(model_path),
        "output_dir": str(output_dir),
        "dataloader": dataloader,
        "calibration_method": args.calibration_method,
        "calibration_num": args.calibration_num,
        "opt_level": args.opt_level,
        "aggressive_partitioning": args.aggressive_partitioning,
        "gen_log": args.gen_log,
    }
    if args.quantization_device:
        kwargs["quantization_device"] = args.quantization_device
    print(f"compiling {model_path} -> {output_dir}")
    dx_com.compile(**kwargs)


def feature_collate(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    p3 = torch.stack([item[0] for item in batch], 0)
    p4 = torch.stack([item[1] for item in batch], 0)
    p5 = torch.stack([item[2] for item in batch], 0)
    return {"p3": p3, "p4": p4, "p5": p5}


def main() -> None:
    args = parse_args()
    try:
        import dx_com
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "dx_com is not installed. Install the DEEPX DX-COM Python wheel or activate the DX-COM environment first."
        ) from exc

    split_dir = Path(args.split_onnx_dir)
    shared_path = split_dir / "shared_backbone.onnx"
    front_path = split_dir / "front_head.onnx"
    top_path = split_dir / "top_head.onnx"
    for path in (shared_path, front_path, top_path):
        if not path.exists():
            raise FileNotFoundError(path)

    out_dir = Path(args.out_dir)
    if not args.skip_shared:
        image_dataset = ImageCalibrationDataset(args.calibration_images, args.calibration_num, args.imgsz)
        image_loader = DataLoader(image_dataset, batch_size=1, shuffle=False)
        compile_model(dx_com, shared_path, out_dir / "shared_backbone", image_loader, args)

    if args.skip_heads:
        print("skipped head compilation")
        return
    cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else out_dir / "feature_cache"
    feature_dataset = FeatureCalibrationDataset(shared_path, args.calibration_images, args.calibration_num, args.imgsz, cache_dir)
    feature_loader = DataLoader(feature_dataset, batch_size=1, shuffle=False, collate_fn=feature_collate)
    compile_model(dx_com, front_path, out_dir / "front_head", feature_loader, args)
    compile_model(dx_com, top_path, out_dir / "top_head", feature_loader, args)

    print("DEEPX compile outputs:")
    print(f"- {out_dir / 'shared_backbone'}")
    print(f"- {out_dir / 'front_head'}")
    print(f"- {out_dir / 'top_head'}")


if __name__ == "__main__":
    main()
