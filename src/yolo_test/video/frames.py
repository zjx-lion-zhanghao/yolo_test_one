"""Video frame extraction utility used during dataset construction."""

from __future__ import annotations

import argparse
from pathlib import Path

from yolo_test.paths import resolve_project_path


def extract_frames(
    video_path: Path,
    output_dir: Path,
    frame_interval: int = 1,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    prefix: str = "frame",
) -> int:
    """Extract frames from a video into image files.

    Args:
        video_path: Source video file.
        output_dir: Destination folder for extracted ``.jpg`` frames.
        frame_interval: Save one frame every N frames. ``1`` saves every frame.
        start_seconds: Start timestamp in seconds.
        end_seconds: Optional end timestamp in seconds.
        prefix: File-name prefix for generated frames.

    Returns:
        Number of frames written.
    """

    import cv2

    if frame_interval < 1:
        raise ValueError("frame_interval must be >= 1")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            raise RuntimeError(f"Could not read FPS from video: {video_path}")

        start_frame = max(0, int(start_seconds * fps))
        end_frame = total_frames if end_seconds is None else min(total_frames, int(end_seconds * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        saved = 0
        frame_index = start_frame
        while frame_index < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_interval == 0:
                timestamp = frame_index / fps
                target = output_dir / f"{prefix}_{frame_index:06d}_{timestamp:.2f}s.jpg"
                cv2.imwrite(str(target), frame)
                saved += 1
            frame_index += 1
        return saved
    finally:
        capture.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("video_path")
    parser.add_argument("output_dir")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--prefix", default="frame")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    count = extract_frames(
        video_path=resolve_project_path(args.video_path),
        output_dir=resolve_project_path(args.output_dir),
        frame_interval=args.interval,
        start_seconds=args.start,
        end_seconds=args.end,
        prefix=args.prefix,
    )
    print(f"Saved {count} frames.")


if __name__ == "__main__":
    main()
