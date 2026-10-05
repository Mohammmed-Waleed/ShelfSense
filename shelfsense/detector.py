"""Thin wrapper around an Ultralytics YOLO model that returns plain Detection objects."""

from __future__ import annotations

import numpy as np

from .types import Detection


class ShelfDetector:
    def __init__(
        self,
        weights: str = "yolo11n.pt",
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str | None = None,
    ):
        from ultralytics import YOLO  # imported lazily: torch is heavy and optional for analytics-only use

        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    @property
    def names(self) -> dict[int, str]:
        return self.model.names

    def detect(self, frame: np.ndarray, track: bool = True) -> list[Detection]:
        """Run inference on a BGR frame. With track=True detections carry persistent track ids."""
        kwargs = dict(conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device, verbose=False)
        if track:
            result = self.model.track(frame, persist=True, tracker="bytetrack.yaml", **kwargs)[0]
        else:
            result = self.model.predict(frame, **kwargs)[0]
        return self._to_detections(result)

    def _to_detections(self, result) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
        return [
            Detection(
                x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
                conf=float(c), cls_id=int(k), label=self.names[int(k)],
                track_id=None if t is None else int(t),
            )
            for b, c, k, t in zip(xyxy, confs, cls_ids, ids)
        ]
