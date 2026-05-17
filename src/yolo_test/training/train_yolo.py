"""Standard training entry point for the local YOLO project.

The script is intentionally thin: Ultralytics remains responsible for the
actual training loop, while this module owns project concerns such as locating
vendored code, placing outputs under ``artifacts/runs``, and documenting the
chosen defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from yolo_test.paths import DEFAULT_RUNS_DIR, PRETRAINED_MODELS, ensure_vendor_on_path, resolve_project_path


def train(
    data: Path,
    model: Path | str,
    name: str,
    epochs: int = 200,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 8,
    device: str = "0",
    optimizer: str = "SGD",
    amp: bool = False,
    cache: bool = False,
) -> None:
    """Launch a YOLO detection training run.

    Args:
        data: Dataset YAML consumed by Ultralytics.
        model: Model weight or model YAML. Paths are resolved before training.
        name: Run name under ``artifacts/runs/detect``.
        epochs: Maximum training epochs.
        imgsz: Square image size passed to Ultralytics.
        batch: Batch size. Tune this according to GPU memory.
        workers: Data-loader workers. On Windows, reduce this if processes fail.
        device: CUDA device id like ``0`` or ``cpu``.
        optimizer: Optimizer name understood by Ultralytics.
        amp: Mixed precision. Disable when a GPU/driver combination is unstable.
        cache: Cache dataset images for faster repeated training.
    """

    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data}")

    ensure_vendor_on_path()
    from ultralytics import YOLO

    model_value = str(resolve_project_path(model)) if isinstance(model, Path) else str(model)
    yolo = YOLO(model_value)
    yolo.train(
        data=str(data),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        workers=workers,
        device=device,
        optimizer=optimizer,
        amp=amp,
        cache=cache,
        project=str(DEFAULT_RUNS_DIR),
        name=name,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO detector using project-standard paths.")
    parser.add_argument("--data", default="configs/datasets/circle_cqu.yaml", help="Dataset YAML path.")
    parser.add_argument(
        "--model",
        default=str(PRETRAINED_MODELS / "yolov8n.pt"),
        help="Model weight or architecture YAML. Defaults to the local official YOLOv8n weight.",
    )
    parser.add_argument("--name", default="circle_yolov8n", help="Run name under artifacts/runs/detect.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cache", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(
        data=resolve_project_path(args.data),
        model=resolve_project_path(args.model),
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        optimizer=args.optimizer,
        amp=args.amp,
        cache=args.cache,
    )


if __name__ == "__main__":
    main()
