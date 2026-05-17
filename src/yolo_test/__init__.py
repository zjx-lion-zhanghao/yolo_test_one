"""Project package for the local YOLO single-class detection workflow.

The package intentionally contains only project-owned code. The upstream
Ultralytics source tree lives under ``vendor/ultralytics`` and is imported
through ``yolo_test.paths.ensure_vendor_on_path`` when a script needs it.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
