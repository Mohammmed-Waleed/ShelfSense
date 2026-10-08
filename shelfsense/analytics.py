"""Turns raw detections into shelf-level insight: counts, empty gaps and alerts.

Pure Python/numpy so it can be tested without a model or a GPU.
"""

from __future__ import annotations

import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .types import Detection


@dataclass
class AnalyticsConfig:
    # Labels that count as shelf stock. Empty means "everything that is not a defect".
    product_classes: list[str] = field(default_factory=list)
    # Labels that always raise an alert (damaged packaging, leaks, wrong item...).
    defect_classes: list[str] = field(default_factory=lambda: ["defect"])
    # Alert when the smoothed count of a label drops below this number.
    low_stock: dict[str, int] = field(default_factory=dict)
    # A horizontal gap wider than gap_factor * median product width is a possible out-of-stock.
    # Below 1.0 so that a single missing product (a hole about one product wide) is caught.
    gap_factor: float = 0.8
    # Products belong to the same shelf row if their centres are within row_tol * median height.
    row_tol: float = 0.6
    # Rows with fewer products than this are too sparse to judge gaps on.
    min_row_items: int = 3
    # Number of frames used for the rolling-median count (kills flicker).
    smooth_frames: int = 5
    # A gap is only reported after it has been seen in this many consecutive frames,
    # so a product the detector misses for a frame or two is not flagged as empty.
    gap_persist_frames: int = 4

    @classmethod
    def from_yaml(cls, path: str | Path) -> AnalyticsConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(f"Unknown analytics config keys: {sorted(unknown)}")
        return cls(**known)


@dataclass(frozen=True)
class Gap:
    x1: float
    y1: float
    x2: float
    y2: float
    row: int


@dataclass(frozen=True)
class Alert:
    kind: str  # "defect" | "low_stock" | "gap"
    message: str

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.message}"


@dataclass
class FrameReport:
    counts: dict[str, int]
    defects: list[Detection]
    gaps: list[Gap]
    alerts: list[Alert]


class ShelfAnalyzer:
    def __init__(self, config: AnalyticsConfig | None = None):
        self.config = config or AnalyticsConfig()
        self._history: deque[Counter] = deque(maxlen=max(1, self.config.smooth_frames))
        self._gap_history: deque[list[Gap]] = deque(maxlen=max(1, self.config.gap_persist_frames))

    def _is_defect(self, det: Detection) -> bool:
        return det.label in self.config.defect_classes

    def _is_product(self, det: Detection) -> bool:
        if self._is_defect(det):
            return False
        pc = self.config.product_classes
        return not pc or det.label in pc

    def analyze(self, detections: list[Detection]) -> FrameReport:
        defects = [d for d in detections if self._is_defect(d)]
        products = [d for d in detections if self._is_product(d)]

        counts = self._smoothed_counts(detections)
        # A damaged item still occupies its slot, so defects count as shelf occupants here.
        gaps = self._persistent_gaps(self.find_gaps(products + defects))

        alerts: list[Alert] = []
        if defects:
            by_label = Counter(d.label for d in defects)
            for label, n in sorted(by_label.items()):
                alerts.append(Alert("defect", f"{n}x {label} detected"))
        for label, minimum in sorted(self.config.low_stock.items()):
            have = counts.get(label, 0)
            if have < minimum:
                alerts.append(Alert("low_stock", f"{label} low: {have}/{minimum}"))
        if gaps:
            alerts.append(Alert("gap", f"{len(gaps)} possible empty slot(s)"))

        return FrameReport(counts=counts, defects=defects, gaps=gaps, alerts=alerts)

    def _smoothed_counts(self, detections: list[Detection]) -> dict[str, int]:
        self._history.append(Counter(d.label for d in detections))
        labels = set().union(*self._history)
        return {
            label: int(statistics.median(c.get(label, 0) for c in self._history))
            for label in sorted(labels)
        }

    def _persistent_gaps(self, gaps: list[Gap]) -> list[Gap]:
        """Keep only gaps that overlap a gap in each of the previous gap_persist_frames - 1 frames."""
        self._gap_history.append(gaps)
        if len(self._gap_history) < self._gap_history.maxlen:
            return []
        past = list(self._gap_history)[:-1]
        return [g for g in gaps if all(any(_same_gap(g, h) for h in frame) for frame in past)]

    def find_gaps(self, products: list[Detection]) -> list[Gap]:
        """Find horizontal holes inside shelf rows where a product is probably missing."""
        if len(products) < self.config.min_row_items:
            return []

        median_h = statistics.median(p.height for p in products)
        rows = self._group_rows(products, self.config.row_tol * median_h)

        gaps: list[Gap] = []
        for row_idx, row in enumerate(rows):
            if len(row) < self.config.min_row_items:
                continue
            row = sorted(row, key=lambda d: d.x1)
            median_w = statistics.median(p.width for p in row)
            y1 = min(p.y1 for p in row)
            y2 = max(p.y2 for p in row)
            right_edge = row[0].x2
            for nxt in row[1:]:
                hole = nxt.x1 - right_edge
                if hole > self.config.gap_factor * median_w:
                    gaps.append(Gap(right_edge, y1, nxt.x1, y2, row_idx))
                right_edge = max(right_edge, nxt.x2)
        return gaps

    @staticmethod
    def _group_rows(products: list[Detection], tolerance: float) -> list[list[Detection]]:
        rows: list[list[Detection]] = []
        row_means: list[float] = []
        for p in sorted(products, key=lambda d: d.center_y):
            if rows and abs(p.center_y - row_means[-1]) <= tolerance:
                rows[-1].append(p)
                row_means[-1] = statistics.fmean(d.center_y for d in rows[-1])
            else:
                rows.append([p])
                row_means.append(p.center_y)
        return rows


def _same_gap(a: Gap, b: Gap) -> bool:
    """Two gaps are the same hole if they overlap vertically and share half the narrower width."""
    if min(a.y2, b.y2) <= max(a.y1, b.y1):
        return False
    overlap = min(a.x2, b.x2) - max(a.x1, b.x1)
    return overlap >= 0.5 * min(a.x2 - a.x1, b.x2 - b.x1)
