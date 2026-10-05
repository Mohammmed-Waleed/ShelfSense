from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One detected object in pixel coordinates (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int
    label: str
    track_id: int | None = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2
