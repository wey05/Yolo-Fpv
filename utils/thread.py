import logging
import time
import threading
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal
from core.detector import ObjectDetector
from core.camera import CameraManager
import numpy as np

logger = logging.getLogger(__name__)


class DetectionThread(QThread):
    """后台检测线程，使用 Qt 信号与主线程通信。"""

    frame_ready = pyqtSignal(np.ndarray, int, float)
    error_occurred = pyqtSignal(str)
    model_switched = pyqtSignal(str)

    def __init__(
        self,
        camera_id: int = 0,
        model_name: str = 'yolov8n.pt',
        resolution: str = '640x480',
    ) -> None:
        super().__init__()
        self.camera_id: int = camera_id
        self.model_name: str = model_name
        self.resolution: str = resolution

        self._lock = threading.Lock()
        self._running: bool = False
        self._conf_threshold: float = 0.5
        self._pending_model: Optional[str] = None
        self._pending_resolution: Optional[str] = None

        self.detector: Optional[ObjectDetector] = None
        self.camera: Optional[CameraManager] = None

    # ── 线程安全的属性访问 ──────────────────────

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @running.setter
    def running(self, value: bool) -> None:
        with self._lock:
            self._running = value

    @property
    def conf_threshold(self) -> float:
        with self._lock:
            return self._conf_threshold

    # ── 主线程调用的控制方法 ─────────────────────

    def set_confidence_threshold(self, threshold: float) -> None:
        """设置检测置信度阈值（线程安全）。"""
        with self._lock:
            self._conf_threshold = max(0.05, min(0.95, threshold))

    def switch_model(self, model_name: str) -> None:
        """请求切换模型（线程安全）。"""
        with self._lock:
            self._pending_model = model_name

    def request_resolution_change(self, resolution: str) -> None:
        """请求切换分辨率（线程安全）。"""
        with self._lock:
            self._pending_resolution = resolution

    def stop(self) -> None:
        """停止检测线程并等待结束。"""
        self.running = False
        self.wait()

    # ── 工作线程逻辑 ──────────────────────────

    def run(self) -> None:
        try:
            self.detector = ObjectDetector(self.model_name)
            self.camera = CameraManager(self.camera_id, self.resolution)

            if not self.camera.open():
                self.error_occurred.emit("无法打开摄像头")
                return

            self.running = True
            fps_counter: int = 0
            fps_timer: float = time.time()
            fps: float = 0.0

            while self.running:
                # 处理挂起的模型切换请求
                with self._lock:
                    pending_model = self._pending_model
                    self._pending_model = None

                if pending_model and self.detector:
                    if self.detector.switch_model(pending_model):
                        self.model_name = pending_model
                        self.model_switched.emit(pending_model)

                # 处理挂起的分辨率切换请求
                with self._lock:
                    pending_res = self._pending_resolution
                    self._pending_resolution = None

                if pending_res and self.camera:
                    self.camera.close()
                    self.camera = CameraManager(self.camera_id, pending_res)
                    if not self.camera.open():
                        self.error_occurred.emit("切换分辨率后无法打开摄像头")
                        return
                    self.resolution = pending_res

                frame = self.camera.read_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                start_time = time.time()
                threshold = self.conf_threshold

                boxes, confidences, class_ids, person_count = self.detector.detect(
                    frame, threshold
                )
                frame_with_boxes = self.detector.draw_boxes(
                    frame, boxes, confidences, class_ids
                )

                fps_counter += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    fps = fps_counter / (now - fps_timer)
                    fps_counter = 0
                    fps_timer = now

                self.frame_ready.emit(frame_with_boxes, person_count, fps)

                elapsed = time.time() - start_time
                sleep_time = max(0.0, (1.0 / 30.0) - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            logger.exception("检测线程错误")
            self.error_occurred.emit(f"检测线程错误: {str(e)}")
        finally:
            if self.camera:
                self.camera.close()
