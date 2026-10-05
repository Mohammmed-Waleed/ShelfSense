"""ShelfSense: real-time product and defect detection on shelves with a fine-tuned YOLO model."""

from .analytics import Alert, AnalyticsConfig, FrameReport, Gap, ShelfAnalyzer
from .types import Detection

__version__ = "0.1.0"

__all__ = [
    "Alert",
    "AnalyticsConfig",
    "Detection",
    "FrameReport",
    "Gap",
    "ShelfAnalyzer",
    "__version__",
]
