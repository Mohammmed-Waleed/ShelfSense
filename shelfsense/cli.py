from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .analytics import AnalyticsConfig, ShelfAnalyzer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shelfsense", description="Real-time shelf product & defect detection.")
    p.add_argument("--version", action="version", version=f"shelfsense {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="generate a synthetic shelf dataset (and optional demo video)")
    s.add_argument("--out", default="datasets/synthetic")
    s.add_argument("--n", type=int, default=240, help="number of images")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--video", default=None, help="also write a demo video to this path")

    t = sub.add_parser("train", help="fine-tune YOLO on a dataset")
    t.add_argument("--data", required=True, help="path to data.yaml")
    t.add_argument("--base", default="yolo11n.pt", help="pretrained checkpoint to fine-tune")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--device", default=None, help="e.g. 0 for first GPU, cpu")
    t.add_argument("--out", default="models/shelfsense.pt", help="where to copy the best weights")

    e = sub.add_parser("eval", help="validate weights on a dataset")
    e.add_argument("--weights", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--imgsz", type=int, default=640)
    e.add_argument("--device", default=None)

    x = sub.add_parser("export", help="export weights for edge deployment")
    x.add_argument("--weights", required=True)
    x.add_argument("--format", default="onnx", help="onnx, openvino, tflite, engine, ...")
    x.add_argument("--imgsz", type=int, default=640)
    x.add_argument("--half", action="store_true")

    d = sub.add_parser("detect", help="run real-time detection on webcam / video / RTSP")
    d.add_argument("--weights", default="models/shelfsense.pt")
    d.add_argument("--source", default="0", help="webcam index, video path or stream URL")
    d.add_argument("--config", default=None, help="analytics YAML (see configs/shelfsense.yaml)")
    d.add_argument("--conf", type=float, default=0.35)
    d.add_argument("--imgsz", type=int, default=640)
    d.add_argument("--device", default=None)
    d.add_argument("--save", default=None, help="write annotated video here")
    d.add_argument("--alert-log", default=None, help="append alerts to this JSONL file")
    d.add_argument("--no-show", action="store_true", help="headless: do not open a window")
    d.add_argument("--max-frames", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "synth":
        from .synthetic import make_dataset, make_demo_video

        yaml_path = make_dataset(args.out, args.n, seed=args.seed)
        print(f"dataset: {yaml_path}")
        if args.video:
            print(f"video:   {make_demo_video(args.video)}")

    elif args.command == "train":
        from .train import train

        best = train(args.data, args.base, args.epochs, args.imgsz, args.batch, args.device, out=args.out)
        print(f"best weights: {best}")

    elif args.command == "eval":
        from .train import evaluate

        print(json.dumps(evaluate(args.weights, args.data, args.imgsz, args.device), indent=2))

    elif args.command == "export":
        from .train import export

        print(f"exported: {export(args.weights, args.format, args.imgsz, args.half)}")

    elif args.command == "detect":
        from .detector import ShelfDetector
        from .realtime import run

        config = AnalyticsConfig.from_yaml(args.config) if args.config else AnalyticsConfig()
        detector = ShelfDetector(args.weights, conf=args.conf, imgsz=args.imgsz, device=args.device)
        summary = run(
            detector, ShelfAnalyzer(config), args.source,
            show=not args.no_show, save=args.save, alert_log=args.alert_log, max_frames=args.max_frames,
        )
        print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
