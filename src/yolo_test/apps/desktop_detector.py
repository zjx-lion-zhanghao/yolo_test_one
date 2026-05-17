"""PySide6 desktop application for camera/video/image YOLO inference."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from yolo_test.paths import PRETRAINED_MODELS, ensure_vendor_on_path, resolve_project_path


os.environ.setdefault("YOLO_VERBOSE", "False")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the YOLO desktop detector.")
    parser.add_argument("--model", default=str(PRETRAINED_MODELS / "yolov8m.pt"))
    return parser


# Keep `--help` usable even before optional GUI/video dependencies are installed.
# The real application imports Qt/OpenCV below; help text only needs argparse.
if any(arg in sys.argv for arg in ("-h", "--help")):
    build_parser().parse_args()
    raise SystemExit(0)


import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class VideoProcessor(QtCore.QThread):
    """Run YOLO inference off the UI thread.

    Camera and video decoding are handled by the main window timer, while this
    worker owns model inference.  That split keeps the UI responsive even when
    a frame takes longer than expected.
    """

    frame_ready = QtCore.Signal(QtGui.QImage)
    failed = QtCore.Signal(str)

    def __init__(self, model_path: Path) -> None:
        super().__init__()
        ensure_vendor_on_path()
        from ultralytics import YOLO

        self.model_path = model_path
        self.model = YOLO(str(model_path))
        self.running = False
        self.current_frame: np.ndarray | None = None
        self.lock = QtCore.QMutex()

    def update_frame(self, frame: np.ndarray) -> None:
        """Copy the latest RGB frame into the worker-owned buffer."""

        self.lock.lock()
        try:
            self.current_frame = frame.copy()
        finally:
            self.lock.unlock()

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        self.running = True
        while self.running:
            self.lock.lock()
            frame = None if self.current_frame is None else self.current_frame.copy()
            self.lock.unlock()

            if frame is None:
                self.msleep(30)
                continue

            try:
                results = self.model(frame)[0]
                annotated = results.plot(line_width=1)
                if not annotated.flags["C_CONTIGUOUS"]:
                    annotated = np.ascontiguousarray(annotated)
                height, width, _ = annotated.shape
                q_img = QtGui.QImage(
                    annotated.data,
                    width,
                    height,
                    3 * width,
                    QtGui.QImage.Format_RGB888,
                ).rgbSwapped()
                self.frame_ready.emit(q_img)
            except Exception as exc:  # noqa: BLE001 - show inference failures in the UI.
                self.failed.emit(str(exc))
                self.running = False
            self.msleep(30)


class MainWindow(QtWidgets.QMainWindow):
    """Main detector window with camera, video, image, and model controls."""

    def __init__(self, model_path: Path) -> None:
        super().__init__()
        self.model_path = model_path
        self.processor = VideoProcessor(model_path)
        self.capture: cv2.VideoCapture | None = None
        self.timer: QtCore.QTimer | None = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        self.setWindowTitle("YOLO 实时检测系统")
        self.resize(1200, 800)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        video_layout = QtWidgets.QHBoxLayout()
        self.original_view = QtWidgets.QLabel()
        self.processed_view = QtWidgets.QLabel()
        for view in (self.original_view, self.processed_view):
            view.setFixedSize(640, 640)
            view.setAlignment(QtCore.Qt.AlignCenter)
            view.setStyleSheet("border: 2px solid #4A90E2; background: #111;")
        video_layout.addWidget(self.original_view)
        video_layout.addWidget(self.processed_view)
        layout.addLayout(video_layout)

        controls = QtWidgets.QHBoxLayout()
        self.camera_button = QtWidgets.QPushButton("摄像头")
        self.video_button = QtWidgets.QPushButton("视频文件")
        self.image_button = QtWidgets.QPushButton("图片识别")
        self.model_button = QtWidgets.QPushButton("选择模型")
        self.stop_button = QtWidgets.QPushButton("停止")
        for button in (
            self.camera_button,
            self.video_button,
            self.image_button,
            self.model_button,
            self.stop_button,
        ):
            button.setFixedSize(120, 40)
            controls.addWidget(button)
        layout.addLayout(controls)

    def _connect_signals(self) -> None:
        self.camera_button.clicked.connect(self.start_camera)
        self.video_button.clicked.connect(self.open_video_file)
        self.image_button.clicked.connect(self.open_image_file)
        self.model_button.clicked.connect(self.select_model_file)
        self.stop_button.clicked.connect(self.stop_all)
        self.processor.frame_ready.connect(self._show_processed_frame)
        self.processor.failed.connect(self._show_error)

    def start_camera(self) -> None:
        self.stop_all()
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self._show_error("无法打开摄像头")
            return
        self._start_timer()
        self.processor.start()

    def open_video_file(self) -> None:
        self.stop_all()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov)")
        if not file_path:
            return
        self.capture = cv2.VideoCapture(file_path)
        if not self.capture.isOpened():
            self._show_error("无法打开视频文件")
            return
        self._start_timer()
        self.processor.start()

    def open_image_file(self) -> None:
        self.stop_all()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片", "", "图像文件 (*.jpg *.jpeg *.png *.bmp)")
        if not file_path:
            return
        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            self._show_error("无法读取图片")
            return

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._show_original_frame(image_rgb)
        self.processor.update_frame(image_rgb)
        if not self.processor.isRunning():
            self.processor.start()

    def select_model_file(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 YOLO 模型权重", "", "PyTorch 模型 (*.pt)")
        if not file_path:
            return
        self.stop_all()
        self.model_path = Path(file_path)
        self.processor = VideoProcessor(self.model_path)
        self.processor.frame_ready.connect(self._show_processed_frame)
        self.processor.failed.connect(self._show_error)
        QtWidgets.QMessageBox.information(self, "模型加载成功", str(self.model_path))

    def _start_timer(self) -> None:
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._capture_frame)
        self.timer.start(30)

    def _capture_frame(self) -> None:
        if self.capture is None or not self.capture.isOpened():
            return
        ok, frame_bgr = self.capture.read()
        if not ok:
            self.stop_all()
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._show_original_frame(frame_rgb)
        self.processor.update_frame(frame_rgb)

    def _show_original_frame(self, frame_rgb: np.ndarray) -> None:
        if not frame_rgb.flags["C_CONTIGUOUS"]:
            frame_rgb = np.ascontiguousarray(frame_rgb)
        height, width, _ = frame_rgb.shape
        q_img = QtGui.QImage(frame_rgb.data, width, height, 3 * width, QtGui.QImage.Format_RGB888)
        self.original_view.setPixmap(QtGui.QPixmap.fromImage(q_img).scaled(
            self.original_view.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))

    def _show_processed_frame(self, q_img: QtGui.QImage) -> None:
        self.processed_view.setPixmap(QtGui.QPixmap.fromImage(q_img).scaled(
            self.processed_view.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))

    def _show_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "错误", message)

    def stop_all(self) -> None:
        if self.timer is not None and self.timer.isActive():
            self.timer.stop()
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        if self.processor.isRunning():
            self.processor.stop()
            self.processor.wait(1000)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_all()
        event.accept()


def main() -> None:
    args = build_parser().parse_args()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(resolve_project_path(args.model))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
