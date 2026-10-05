"""Synthetic shelf images + YOLO labels, so the full pipeline can be run without downloading a dataset.

Classes: 0 = product, 1 = defect (a crushed/cracked product). Gaps are simply missing products.
It is a smoke-test / demo dataset: train on real shelf photos (see README) for real-world use.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

CLASS_NAMES = ["product", "defect"]
SIZE = 640
ROWS = 4
ROW_H = SIZE // ROWS


@dataclass
class Item:
    x1: int
    y1: int
    x2: int
    y2: int
    color: tuple[int, int, int]
    defect: bool = False


def make_layout(rng: random.Random, gap_prob: float = 0.12, defect_prob: float = 0.07) -> list[Item]:
    items: list[Item] = []
    for r in range(ROWS):
        floor = (r + 1) * ROW_H - 14
        x = rng.randint(8, 30)
        while True:
            w, h = rng.randint(42, 66), rng.randint(70, 112)
            if x + w > SIZE - 8:
                break
            roll = rng.random()
            if roll < gap_prob:
                x += w + rng.randint(2, 8)  # missing product -> empty slot
                continue
            color = tuple(rng.randint(40, 235) for _ in range(3))
            items.append(Item(x, floor - h, x + w, floor, color, defect=roll > 1 - defect_prob))
            x += w + rng.randint(2, 8)
    return items


def render(items: list[Item], rng: random.Random) -> np.ndarray:
    img = np.full((SIZE, SIZE, 3), 215, np.uint8)
    for r in range(ROWS):
        y = (r + 1) * ROW_H - 14
        cv2.rectangle(img, (0, y), (SIZE, y + 14), (70, 90, 120), -1)
    for it in items:
        cv2.rectangle(img, (it.x1, it.y1), (it.x2, it.y2), it.color, -1)
        band_h = (it.y2 - it.y1) // 4
        band_y = it.y1 + (it.y2 - it.y1) // 3
        cv2.rectangle(img, (it.x1 + 4, band_y), (it.x2 - 4, band_y + band_h), (245, 245, 245), -1)
        cv2.rectangle(img, (it.x1, it.y1), (it.x2, it.y2), (30, 30, 30), 1)
        if it.defect:  # dark crack lines + dent
            cv2.line(img, (it.x1, it.y1 + 8), (it.x2, it.y2 - 8), (20, 20, 20), 2)
            cv2.line(img, (it.x2, it.y1 + 14), (it.x1 + 6, it.y2 - 4), (20, 20, 20), 2)
            cv2.circle(img, ((it.x1 + it.x2) // 2, it.y1 + 10), 5, (0, 0, 200), -1)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 4, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def yolo_lines(items: list[Item]) -> list[str]:
    lines = []
    for it in items:
        cx, cy = (it.x1 + it.x2) / 2 / SIZE, (it.y1 + it.y2) / 2 / SIZE
        w, h = (it.x2 - it.x1) / SIZE, (it.y2 - it.y1) / SIZE
        lines.append(f"{int(it.defect)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def make_dataset(out_dir: str | Path, n_images: int = 240, val_fraction: float = 0.2, seed: int = 0) -> Path:
    """Write a YOLO-format dataset and return the path to its data.yaml."""
    out = Path(out_dir).resolve()
    rng = random.Random(seed)
    n_val = max(1, int(n_images * val_fraction))
    for split, n in (("train", n_images - n_val), ("val", n_val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            items = make_layout(rng)
            cv2.imwrite(str(out / "images" / split / f"{split}_{i:04d}.jpg"), render(items, rng))
            (out / "labels" / split / f"{split}_{i:04d}.txt").write_text("\n".join(yolo_lines(items)) + "\n")
    data = {"path": str(out), "train": "images/train", "val": "images/val", "names": dict(enumerate(CLASS_NAMES))}
    yaml_path = out / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return yaml_path


def make_demo_video(out_path: str | Path, seconds: int = 12, fps: int = 10, seed: int = 1) -> Path:
    """A shelf that gets restocked/emptied over time: products vanish (gaps) and defects appear."""
    rng = random.Random(seed)
    base = make_layout(rng, gap_prob=0.0, defect_prob=0.0)
    removal_order = rng.sample(range(len(base)), k=len(base))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (SIZE, SIZE))
    total = seconds * fps
    for f in range(total):
        removed = set(removal_order[: int(len(base) * 0.35 * f / total)])
        items = [Item(i.x1, i.y1, i.x2, i.y2, i.color, defect=(idx % 11 == 0 and f > total // 2))
                 for idx, i in enumerate(base) if idx not in removed]
        writer.write(render(items, random.Random(f)))
    writer.release()
    return out_path
