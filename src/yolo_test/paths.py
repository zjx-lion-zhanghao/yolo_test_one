"""Shared path helpers for the project.

Keeping all important paths in one module avoids a common YOLO-project problem:
individual scripts quietly hard-code their own Windows desktop paths.  The
functions below make scripts runnable from any current working directory while
still allowing command-line arguments to override every data/model path.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""Absolute path to the repository root."""

SRC_DIR = PROJECT_ROOT / "src"
VENDOR_ULTRALYTICS = PROJECT_ROOT / "vendor" / "ultralytics"
PRETRAINED_MODELS = PROJECT_ROOT / "models" / "pretrained"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_RUNS_DIR = ARTIFACTS_DIR / "runs"


def resolve_project_path(value: str | Path) -> Path:
    """Return an absolute path, resolving relative paths from project root.

    Args:
        value: User-provided path. Relative values are interpreted as
            project-relative so commands behave consistently from terminals,
            IDE run configurations, and CI jobs.

    Returns:
        Absolute ``Path`` object.
    """

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_vendor_on_path() -> None:
    """Make the vendored Ultralytics source importable.

    The project keeps upstream code in ``vendor/ultralytics`` instead of mixing
    local scripts into the official repository.  Adding the vendor path at
    runtime lets ``from ultralytics import YOLO`` keep working without forcing
    every user to run ``pip install -e vendor/ultralytics`` first.
    """

    vendor = str(VENDOR_ULTRALYTICS)
    source = str(SRC_DIR)
    if source not in sys.path:
        sys.path.insert(0, source)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
