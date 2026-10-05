"""Real-time loop: webcam / video file / RTSP stream -> detections -> analytics -> overlay."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .analytics import FrameReport, ShelfAnalyzer
from .types import Detection

GREEN, RED, AMBER, WHITE, BLACK = (80, 200, 80), (60, 60, 230), (0, 170, 255), (255, 255, 255), (0, 0, 0)


class AlertLog:
    """Appends alerts to a JSONL file, suppressing repeats of the same alert within `cooldown` seconds."""

    def __init__(self, path: str | Path | None, cooldown: float = 10.0):
        self.path = Path(path) if path else None
        self.cooldown = cooldown
        self._last: dict[str, float] = {}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, report: FrameReport, frame_idx: int) -> None:
        if not self.path:
            return
        now = time.time()
        for alert in report.alerts:
            if now - self._last.get(alert.key, 0.0) < self.cooldown:
                continue
            self._last[alert.key] = now
            row = {"ts": now, "frame": frame_idx, "kind": alert.kind, "message": alert.message}
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")


def draw(frame: np.ndarray, detections: list[Detection], report: FrameReport, fps: float, defect_labels: set[str]) -> np.ndarray:
    out = frame.copy()
    for g in report.gaps:
        p1, p2 = (int(g.x1), int(g.y1)), (int(g.x2), int(g.y2))
        cv2.rectangle(out, p1, p2, AMBER, 2, cv2.LINE_AA)
        cv2.putText(out, "gap", (p1[0] + 3, p1[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMBER, 1, cv2.LINE_AA)
    for d in detections:
        color = RED if d.label in defect_labels else GREEN
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, 2, cv2.LINE_AA)
        tag = f"{d.label} {d.conf:.2f}" + (f" #{d.track_id}" if d.track_id is not None else "")
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(out, tag, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLACK, 1, cv2.LINE_AA)

    lines = [f"FPS {fps:4.1f}"] + [f"{k}: {v}" for k, v in report.counts.items()] + [f"! {a.message}" for a in report.alerts]
    panel_h = 8 + 20 * len(lines)
    cv2.rectangle(out, (0, 0), (260, panel_h), BLACK, -1)
    for i, text in enumerate(lines):
        color = AMBER if text.startswith("!") else WHITE
        cv2.putText(out, text, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def parse_source(source: str) -> int | str:
    """'0' -> webcam index 0; anything else is a file path or stream URL."""
    return int(source) if source.isdigit() else source


def run(
    detector,
    analyzer: ShelfAnalyzer,
    source: str = "0",
    show: bool = True,
    save: str | None = None,
    alert_log: str | None = None,
    max_frames: int | None = None,
) -> dict:
    cap = cv2.VideoCapture(parse_source(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")

    writer = None
    log = AlertLog(alert_log)
    defect_labels = set(analyzer.config.defect_classes)
    fps, frames, alerts_seen = 0.0, 0, 0
    try:
        while max_frames is None or frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            detections = detector.detect(frame, track=True)
            report = analyzer.analyze(detections)
            dt = time.perf_counter() - t0
            fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt

            log.record(report, frames)
            alerts_seen += len(report.alerts)
            annotated = draw(frame, detections, report, fps, defect_labels)

            if save:
                if writer is None:
                    h, w = annotated.shape[:2]
                    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    Path(save).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
                writer.write(annotated)
            if show:
                cv2.imshow("ShelfSense (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frames += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    return {"frames": frames, "fps": round(fps, 1), "alerts": alerts_seen}
