# YOLO Optim

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jje-seok/Yolo_Fine_tuning/blob/main/notebooks/deepx_dual_yolov8_finetune_colab.ipynb)

Dual-camera YOLOv8n fine-tuning project for a DEEPX deployment flow.

## What This Repository Contains

- Shared YOLOv8n backbone with separate `front_head` and `top_head`
- Roboflow + real-site calibration fine-tuning script
- Manual YOLO label review tool for person class labels
- Split ONNX export for `shared_backbone.onnx`, `front_head.onnx`, and `top_head.onnx`
- DEEPX compile helper scripts

## Quick Start

Open the Colab notebook directly:

```text
notebooks/deepx_dual_yolov8_finetune_colab.ipynb
```

or use the Colab badge at the top of this README.

When opened from GitHub/Colab, the notebook automatically clones or updates this
repository into:

```text
/content/Yolo_Fine_tuning
```

The notebook also needs these runtime assets, which are intentionally not stored
in git:

```text
YOLOV8N.onnx
calibration/labeled/front
calibration/labeled/top
```

Put them in one of these Google Drive folders before running the setup cell:

```text
/content/drive/MyDrive/YOLO_optim
/content/drive/MyDrive/Yolo_Fine_tuning
/content/drive/MyDrive/Yolo_Fine_tuning_assets
/content/drive/MyDrive/YOLO_optim_assets
```

or set `YOLO_OPTIM_ASSET_ROOT` to the folder that contains `YOLOV8N.onnx` and
`calibration/labeled`.

Large datasets, checkpoints, ONNX files, DXNN files, calibration images, and DEEPX manuals are intentionally excluded from git.

## Main Commands

Site calibration fine-tuning:

```bash
python -m dual_yolov8.site_finetune \
  --onnx YOLOV8N.onnx \
  --front-roboflow-root datasets/roboflow_raw/person-detection-3 \
  --top-roboflow-root datasets/roboflow_raw/CCTV-Indoor-Person-Detection-1 \
  --front-calib-root calibration/labeled/front \
  --top-calib-root calibration/labeled/top \
  --out runs/dual_yolov8_site_mix
```

Manual label review:

```bash
python -m dual_yolov8.review_labels --dataset calibration/labeled/front
python -m dual_yolov8.review_labels --dataset calibration/labeled/top
```

More detailed documentation is in [dual_yolov8_README.md](dual_yolov8_README.md).
