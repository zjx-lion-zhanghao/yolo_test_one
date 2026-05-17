"""Convert Pascal VOC XML annotations to YOLO TXT labels.

This module replaces the earlier one-off ``xmlchange.py`` script with a
parameterized, repeatable converter.  It validates image sizes, skips unknown
classes by default, and keeps the class list explicit so training labels remain
stable across machines and annotation batches.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from yolo_test.paths import resolve_project_path


@dataclass(frozen=True)
class ConvertSummary:
    """Small immutable summary returned after conversion."""

    xml_files: int
    labels_written: int
    boxes_written: int
    boxes_skipped: int


def _read_required_int(parent: ET.Element, tag: str, xml_path: Path) -> int:
    """Read an integer XML child and raise a clear error when it is missing."""

    child = parent.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"{xml_path}: missing <{tag}>")
    return int(float(child.text))


def _convert_box(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert VOC corner coordinates to normalized YOLO center coordinates."""

    box_width = xmax - xmin
    box_height = ymax - ymin
    x_center = xmin + box_width / 2.0
    y_center = ymin + box_height / 2.0
    return (
        x_center / image_width,
        y_center / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def convert_voc_xml_folder(
    xml_dir: Path,
    output_dir: Path,
    classes: list[str],
    skip_unknown_classes: bool = True,
) -> ConvertSummary:
    """Convert all ``*.xml`` files in a folder to YOLO label files.

    Args:
        xml_dir: Folder containing Pascal VOC XML files.
        output_dir: Destination folder for YOLO ``.txt`` labels.
        classes: Ordered class names. The index in this list is the YOLO class
            id, so changing the order changes the meaning of every label.
        skip_unknown_classes: When true, objects whose names are not in
            ``classes`` are ignored. When false, conversion stops with an error.

    Returns:
        ``ConvertSummary`` with file and box counts for logging or tests.
    """

    if not xml_dir.is_dir():
        raise FileNotFoundError(f"XML folder does not exist: {xml_dir}")
    if not classes:
        raise ValueError("At least one class name is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    class_to_id = {name: index for index, name in enumerate(classes)}

    xml_files = 0
    labels_written = 0
    boxes_written = 0
    boxes_skipped = 0

    for xml_path in sorted(xml_dir.glob("*.xml")):
        xml_files += 1
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        if size is None:
            raise ValueError(f"{xml_path}: missing <size>")

        image_width = _read_required_int(size, "width", xml_path)
        image_height = _read_required_int(size, "height", xml_path)
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"{xml_path}: invalid image size {image_width}x{image_height}")

        yolo_lines: list[str] = []
        for obj in root.findall("object"):
            name_node = obj.find("name")
            class_name = name_node.text.strip() if name_node is not None and name_node.text else ""
            if class_name not in class_to_id:
                boxes_skipped += 1
                if skip_unknown_classes:
                    continue
                raise ValueError(f"{xml_path}: unknown class {class_name!r}")

            bndbox = obj.find("bndbox")
            if bndbox is None:
                raise ValueError(f"{xml_path}: object {class_name!r} is missing <bndbox>")

            xmin = _read_required_int(bndbox, "xmin", xml_path)
            ymin = _read_required_int(bndbox, "ymin", xml_path)
            xmax = _read_required_int(bndbox, "xmax", xml_path)
            ymax = _read_required_int(bndbox, "ymax", xml_path)

            x_center, y_center, width, height = _convert_box(
                xmin, ymin, xmax, ymax, image_width, image_height
            )
            yolo_lines.append(
                f"{class_to_id[class_name]} "
                f"{x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"
            )
            boxes_written += 1

        (output_dir / f"{xml_path.stem}.txt").write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
            encoding="utf-8",
        )
        labels_written += 1

    return ConvertSummary(xml_files, labels_written, boxes_written, boxes_skipped)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML annotations to YOLO labels.")
    parser.add_argument("--xml-dir", required=True, help="Folder containing VOC XML annotations.")
    parser.add_argument("--output-dir", required=True, help="Folder where YOLO .txt labels will be written.")
    parser.add_argument("--classes", required=True, help="Comma-separated class names, in YOLO id order.")
    parser.add_argument(
        "--fail-on-unknown",
        action="store_true",
        help="Raise an error instead of skipping objects whose class name is not configured.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = convert_voc_xml_folder(
        xml_dir=resolve_project_path(args.xml_dir),
        output_dir=resolve_project_path(args.output_dir),
        classes=[item.strip() for item in args.classes.split(",") if item.strip()],
        skip_unknown_classes=not args.fail_on_unknown,
    )
    print(summary)


if __name__ == "__main__":
    main()
