"""Safe file-maintenance commands for image/label datasets.

The old project had several deletion and renaming scripts with hard-coded
folders.  This module keeps those operations behind explicit command-line
arguments and supports ``--dry-run`` so a user can inspect the planned changes
before touching a dataset.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from yolo_test.paths import resolve_project_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def remove_empty_files(folder: Path, dry_run: bool = True) -> list[Path]:
    """Delete zero-byte files recursively and return the affected paths."""

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    affected: list[Path] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.stat().st_size == 0:
            affected.append(path)
            if not dry_run:
                path.unlink()
    return affected


def remove_unmatched_by_stem(source: Path, reference: Path, dry_run: bool = True) -> list[Path]:
    """Delete files in ``source`` whose stem does not exist in ``reference``.

    This is useful for keeping ``images`` and ``labels`` synchronized.  The
    function compares only file stems, so ``frame_001.jpg`` matches
    ``frame_001.txt``.
    """

    if not source.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {source}")
    if not reference.is_dir():
        raise FileNotFoundError(f"Reference folder does not exist: {reference}")

    reference_stems = {path.stem for path in reference.iterdir() if path.is_file()}
    affected: list[Path] = []
    for path in sorted(source.iterdir()):
        if path.is_file() and path.stem not in reference_stems:
            affected.append(path)
            if not dry_run:
                path.unlink()
    return affected


def organize_xml_and_images(folder: Path, dry_run: bool = True) -> list[tuple[Path, Path]]:
    """Move XML files to ``xml/`` and images to ``jpg/`` under ``folder``."""

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    moves: list[tuple[Path, Path]] = []
    xml_dir = folder / "xml"
    image_dir = folder / "jpg"

    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".xml":
            target = xml_dir / path.name
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            target = image_dir / path.name
        else:
            continue
        moves.append((path, target))
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))

    return moves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe dataset file-maintenance utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    empty = subparsers.add_parser("remove-empty", help="Remove zero-byte files recursively.")
    empty.add_argument("--folder", required=True)
    empty.add_argument("--apply", action="store_true", help="Actually modify files. Default is dry-run.")

    unmatched = subparsers.add_parser("remove-unmatched", help="Remove source files missing a reference stem.")
    unmatched.add_argument("--source", required=True)
    unmatched.add_argument("--reference", required=True)
    unmatched.add_argument("--apply", action="store_true", help="Actually modify files. Default is dry-run.")

    organize = subparsers.add_parser("organize", help="Move XML and image files into separate folders.")
    organize.add_argument("--folder", required=True)
    organize.add_argument("--apply", action="store_true", help="Actually modify files. Default is dry-run.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dry_run = not args.apply

    if args.command == "remove-empty":
        affected = remove_empty_files(resolve_project_path(args.folder), dry_run=dry_run)
        for path in affected:
            print(path)
        print(f"{'Would remove' if dry_run else 'Removed'} {len(affected)} empty files.")
    elif args.command == "remove-unmatched":
        affected = remove_unmatched_by_stem(
            resolve_project_path(args.source),
            resolve_project_path(args.reference),
            dry_run=dry_run,
        )
        for path in affected:
            print(path)
        print(f"{'Would remove' if dry_run else 'Removed'} {len(affected)} unmatched files.")
    elif args.command == "organize":
        moves = organize_xml_and_images(resolve_project_path(args.folder), dry_run=dry_run)
        for source, target in moves:
            print(f"{source} -> {target}")
        print(f"{'Would move' if dry_run else 'Moved'} {len(moves)} files.")


if __name__ == "__main__":
    main()
