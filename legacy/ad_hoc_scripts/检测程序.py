from PySide6 import QtWidgets, QtCore, QtGui
import cv2
import os
import numpy as np
from ultralytics import YOLO

os.environ['YOLO_VERBOSE'] = 'False'  # 禁用YOLO调试输出


class VideoProcessor(QtCore.QThread):
    frame_ready = QtCore.Signal(QtGui.QImage)2
    finished = QtCore.Signal()

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.running = False
        self.current_frame = None
        self.lock = QtCore.QMutex()

    def update_frame(self, frame):
        self.lock.lock()
        self.current_frame = frame.copy()
        self.lock.unlock()

    def run(self):
        self.running = True
        while self.running:
            if self.current_frame is not None:
                self.lock.lock()
                frame = self.current_frame
                self.lock.unlock()

                results = self.model(frame)[0]
                annotated_frame = results.plot(line_width=1)

                if not annotated_frame.flags['C_CONTIGUOUS']:
                    annotated_frame = np.ascontiguousarray(annotated_frame)

                height, width, _ = annotated_frame.shape
                q_img = QtGui.QImage(annotated_frame.data, width, height, 3 * width,
                                     QtGui.QImage.Format_RGB888).rgbSwapped()
                self.frame_ready.emit(q_img)
            self.msleep(30)
        self.finished.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_path = 'yolov8m.pt'  # 替换为你自己的模型
        self.video_processor = VideoProcessor(self.model_path)
        self.cap = None
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        self.setWindowTitle("YOLOv8实时检测系统")
        self.resize(1200, 800)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        video_layout = QtWidgets.QHBoxLayout()
        self.original_view = QtWidgets.QLabel()
        self.processed_view = QtWidgets.QLabel()
        for view in [self.original_view, self.processed_view]:
            view.setFixedSize(640,640)
            view.setStyleSheet("border: 2px solid #4A90E2;")
        video_layout.addWidget(self.original_view)
        video_layout.addWidget(self.processed_view)
        main_layout.addLayout(video_layout)

        control_layout = QtWidgets.QHBoxLayout()
        self.btn_camera = QtWidgets.QPushButton("摄像头")
        self.btn_video = QtWidgets.QPushButton("视频文件")
        self.btn_image = QtWidgets.QPushButton("图片识别")
        self.btn_stop = QtWidgets.QPushButton("停止")
        self.btn_select_model = QtWidgets.QPushButton("选择模型")
        for btn in [self.btn_camera, self.btn_video, self.btn_image, self.btn_stop, self.btn_select_model]:
            btn.setFixedSize(120, 40)
            btn.setStyleSheet("font-size: 14px;")
            control_layout.addWidget(btn)
        main_layout.addLayout(control_layout)

    def setup_connections(self):
        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_video.clicked.connect(self.open_video_file)
        self.btn_image.clicked.connect(self.select_image)
        self.btn_stop.clicked.connect(self.stop_all)
        self.btn_select_model.clicked.connect(self.select_model_file)
        self.video_processor.frame_ready.connect(self.update_processed_view)
        self.video_processor.finished.connect(self.cleanup)

    def start_camera(self):
        self.stop_all()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if self.cap.isOpened():
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.capture_frame)
            self.timer.start(30)
            self.video_processor.start()

    def open_video_file(self):
        self.stop_all()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.capture_frame)
            self.timer.start(30)
            self.video_processor.start()

    def select_image(self):
        self.stop_all()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择图片", "", "图像文件 (*.jpg *.png *.bmp)")
        if file_path:
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                QtWidgets.QMessageBox.critical(self, "错误", "无法读取图片")
                return

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if not img_rgb.flags['C_CONTIGUOUS']:
                img_rgb = np.ascontiguousarray(img_rgb)

            q_img_orig = QtGui.QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0],
                                      3 * img_rgb.shape[1], QtGui.QImage.Format_RGB888)
            self.original_view.setPixmap(QtGui.QPixmap.fromImage(q_img_orig))

            try:
                model = YOLO(self.model_path)
                results = model(img_rgb)[0]
                annotated = results.plot(line_width=1)

                if not annotated.flags['C_CONTIGUOUS']:
                    annotated = np.ascontiguousarray(annotated)

                q_img_annot = QtGui.QImage(annotated.data, annotated.shape[1], annotated.shape[0],
                                           3 * annotated.shape[1], QtGui.QImage.Format_RGB888).rgbSwapped()
                self.processed_view.setPixmap(QtGui.QPixmap.fromImage(q_img_annot))
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "识别失败", f"错误: {str(e)}")

    def capture_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not frame_rgb.flags['C_CONTIGUOUS']:
                frame_rgb = np.ascontiguousarray(frame_rgb)

            h, w, _ = frame_rgb.shape
            q_img = QtGui.QImage(frame_rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            self.original_view.setPixmap(QtGui.QPixmap.fromImage(q_img))
            self.video_processor.update_frame(frame_rgb)

    def update_processed_view(self, q_img):
        self.processed_view.setPixmap(QtGui.QPixmap.fromImage(q_img))

    def stop_all(self):
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.video_processor.running = False

    def cleanup(self):
        self.original_view.clear()
        self.processed_view.clear()

    def select_model_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择YOLO模型权重文件", "", "PyTorch 模型 (*.pt)")
        if file_path:
            self.stop_all()
            try:
                self.model_path = file_path
                self.video_processor = VideoProcessor(self.model_path)
                self.video_processor.frame_ready.connect(self.update_processed_view)
                self.video_processor.finished.connect(self.cleanup)
                QtWidgets.QMessageBox.information(self, "模型加载成功", f"已加载模型: {file_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "模型加载失败", f"错误: {str(e)}")

    def closeEvent(self, event):
        self.stop_all()
        self.video_processor.quit()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    window = MainWindow()
    window.show()
    app.exec()
