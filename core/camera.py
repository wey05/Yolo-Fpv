import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 预设分辨率选项
RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "640x480": (640, 480),
    "1280x720": (1280, 720),
    "1920x1080": (1920, 1080),
}


class CameraManager:
    """摄像头管理器，支持设备枚举、分辨率切换。"""

    def __init__(self, camera_id: int = 0, resolution: str = "640x480") -> None:
        self.camera_id: int = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened: bool = False

        w, h = RESOLUTION_PRESETS.get(resolution, (640, 480))
        self.frame_width: int = w
        self.frame_height: int = h
        self.fps: int = 30

    def open(self) -> bool:
        """打开摄像头并设置参数。"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                logger.warning("无法打开摄像头 %d", self.camera_id)
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            self.is_opened = True
            logger.info(
                "摄像头已打开: %dx%d @ %dfps",
                self.frame_width, self.frame_height, self.fps,
            )
            return True

        except Exception as e:
            logger.error("摄像头初始化错误: %s", e)
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧图像。"""
        if not self.is_opened or self.cap is None:
            return None

        ret, frame = self.cap.read()

        if not ret:
            logger.debug("读取帧失败")
            return None

        return frame

    def close(self) -> None:
        """释放摄像头资源。"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False
        logger.info("摄像头已关闭")

    def get_camera_info(self) -> dict:
        """获取当前摄像头信息。"""
        return {
            'camera_id': self.camera_id,
            'width': self.frame_width,
            'height': self.frame_height,
            'fps': self.fps,
            'is_opened': self.is_opened,
        }

    @staticmethod
    def enumerate_cameras(max_test: int = 5) -> list[int]:
        """枚举系统中可用的摄像头设备 ID。"""
        available: list[int] = []
        for i in range(max_test):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append(i)
                cap.release()
        logger.info("检测到 %d 个可用摄像头: %s", len(available), available)
        return available
