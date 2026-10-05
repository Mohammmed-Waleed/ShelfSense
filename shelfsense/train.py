"""Fine-tune a pretrained YOLO model on a shelf dataset, evaluate it and export it."""

from __future__ import annotations

import shutil
from pathlib import Path


def train(
    data: str,
    base: str = "yolo11n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    project: str = "runs",
    name: str = "shelfsense",
    out: str = "models/shelfsense.pt",
) -> Path:
    from ultralytics import YOLO

    model = YOLO(base)
    model.train(
        data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=device,
        project=str(Path(project).resolve()), name=name, exist_ok=True,
        patience=20, cos_lr=True, close_mosaic=10, plots=True,
        # Shelves are photographed upright: no vertical flips, only mild rotation.
        flipud=0.0, fliplr=0.5, degrees=5.0, hsv_v=0.4,
    )
    best = Path(model.trainer.best)
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest)
    return dest


def evaluate(weights: str, data: str, imgsz: int = 640, device: str | None = None) -> dict[str, float]:
    from ultralytics import YOLO

    m = YOLO(weights).val(data=data, imgsz=imgsz, device=device, plots=False, verbose=False)
    return {
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
    }


def export(weights: str, fmt: str = "onnx", imgsz: int = 640, half: bool = False) -> str:
    """Export for edge deployment (onnx, openvino, tflite, engine...)."""
    from ultralytics import YOLO

    return str(YOLO(weights).export(format=fmt, imgsz=imgsz, half=half))
