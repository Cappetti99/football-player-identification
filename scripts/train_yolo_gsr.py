"""Train or fine-tune YOLO on a SoccerNet-GSR YOLO-format dataset.

This expects the existing conversion output:
    /media/data-lie/cappetti/dataset/soccernet_gsr_yolo/data.yaml
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO on SoccerNet-GSR YOLO labels.")
    parser.add_argument("--data-yaml", default="/media/data-lie/cappetti/dataset/soccernet_gsr_yolo/data.yaml")
    parser.add_argument("--base-model", default="yolo26x.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/ft_yolo_gsr")
    parser.add_argument("--name", default="yolo26x_gsr")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics before training") from exc
    if not Path(args.data_yaml).exists():
        raise FileNotFoundError(args.data_yaml)
    model = YOLO(args.base_model)
    train_args = {
        "data": args.data_yaml,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
    }
    if args.device is not None:
        train_args["device"] = args.device
    model.train(**train_args)


if __name__ == "__main__":
    main()

