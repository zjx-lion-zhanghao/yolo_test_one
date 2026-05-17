"""Create a YOLO train/val/test folder layout from image and label folders."""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from yolo_test.paths import resolve_project_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SplitSummary:
    train: int
    val: int
    test: int
    unmatched_images: int
    unmatched_labels: int


def _copy_pairs(pairs: list[tuple[Path, Path]], output_root: Path, split_name: str) -> None:
    """Copy matched image/label pairs into the canonical YOLO split layout."""

    image_dir = output_root / split_name / "images"
    label_dir = output_root / split_name / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for image_path, label_path in pairs:
        shutil.copy2(image_path, image_dir / image_path.name)
        shutil.copy2(label_path, label_dir / label_path.name)


def split_dataset(
    image_dir: Path,
    label_dir: Path,
    output_root: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
) -> SplitSummary:
    """Split matched image/label pairs into train, val, and test folders.

    The function only copies files. It never deletes source data, which makes it
    safe to run repeatedly while experimenting with split ratios.
    """

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label folder does not exist: {label_dir}")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    labels = {
        path.stem: path
        for path in label_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    }

    matched_keys = sorted(set(images) & set(labels))
    unmatched_images = len(set(images) - set(labels))
    unmatched_labels = len(set(labels) - set(images))
    pairs = [(images[key], labels[key]) for key in matched_keys]

    rng = random.Random(seed)
    rng.shuffle(pairs)

    train_end = int(len(pairs) * train_ratio)
    val_end = train_end + int(len(pairs) * val_ratio)
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    _copy_pairs(train_pairs, output_root, "train")
    _copy_pairs(val_pairs, output_root, "val")
    _copy_pairs(test_pairs, output_root, "test")

    return SplitSummary(
        train=len(train_pairs),
        val=len(val_pairs),
        test=len(test_pairs),
        unmatched_images=unmatched_images,
        unmatched_labels=unmatched_labels,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split images and YOLO labels into train/val/test folders.")
    parser.add_argument("--images", required=True, help="Source image folder.")
    parser.add_argument("--labels", required=True, help="Source YOLO label folder.")
    parser.add_argument("--output", required=True, help="Output dataset root.")
    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic shuffle seed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = split_dataset(
        image_dir=resolve_project_path(args.images),
        label_dir=resolve_project_path(args.labels),
        output_root=resolve_project_path(args.output),
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )
    print(summary)


if __name__ == "__main__":
    main()
