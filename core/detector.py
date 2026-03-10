import logging
from typing import Optional

import torch
from ultralytics import YOLO
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _generate_color_palette(num_classes: int = 80) -> dict[int, tuple[int, ...]]:
    """预计算所有类别的固定颜色映射，避免每帧重复 seed 随机数。"""
    palette: dict[int, tuple[int, ...]] = {}
    rng = np.random.RandomState(42)
    for cls_id in range(num_classes):
        color = tuple(int(c) for c in rng.randint(50, 255, 3))
        palette[cls_id] = color
    return palette


class ObjectDetector:
    """YOLO 目标检测器，支持多类别检测和运行时模型切换。"""

    DETECT_ALL_CLASSES: bool = True
    _COLOR_PALETTE: dict[int, tuple[int, ...]] = _generate_color_palette(200)

    def __init__(self, model_name: str = 'yolov8n.pt') -> None:
        self.model_name: str = model_name
        self.model: Optional[YOLO] = None
        self.device: str = 'cpu'
        self.class_names: dict[int, str] = {}
        self._initialize_model()

    def _initialize_model(self) -> None:
        try:
            if torch.cuda.is_available():
                self.device = 'cuda'
                logger.info("使用GPU加速: %s", torch.cuda.get_device_name(0))
            else:
                self.device = 'cpu'
                logger.info("使用CPU模式")

            logger.info("正在加载模型: %s", self.model_name)
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            self.class_names = self.model.names
            logger.info("模型加载成功，共 %d 个类别", len(self.class_names))

        except Exception as e:
            logger.error("模型加载失败: %s", e)
            raise

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5,
    ) -> tuple[list[list[int]], list[float], list[int], int]:
        """
        对单帧图像执行目标检测。

        Returns:
            (boxes, confidences, class_ids, person_count)
        """
        if self.model is None:
            return [], [], [], 0

        try:
            results = self.model(frame, conf=conf_threshold, verbose=False, device=self.device)

            boxes: list[list[int]] = []
            confidences: list[float] = []
            class_ids: list[int] = []
            person_count: int = 0

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])

                    if not self.DETECT_ALL_CLASSES and cls != 0:
                        continue

                    if cls == 0:
                        person_count += 1

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    confidences.append(conf)
                    class_ids.append(cls)

            return boxes, confidences, class_ids, person_count

        except Exception as e:
            logger.error("检测错误: %s", e)
            return [], [], [], 0

    def draw_boxes(
        self,
        frame: np.ndarray,
        boxes: list[list[int]],
        confidences: list[float],
        class_ids: list[int],
    ) -> np.ndarray:
        """在帧上绘制检测框和标签。"""
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box

            if self.DETECT_ALL_CLASSES:
                color = self._COLOR_PALETTE.get(cls_id, (200, 200, 200))
                class_name = self.class_names.get(cls_id, str(cls_id))
                label = f"{class_name} {conf:.2f}"
            else:
                if conf > 0.8:
                    color = (0, 255, 0)
                elif conf > 0.6:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)
                label = f"Person {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_w, label_h = label_size

            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

        return frame

    def switch_model(self, model_name: str) -> bool:
        """运行时切换 YOLO 模型。"""
        if model_name == self.model_name:
            return False
        try:
            logger.info("正在切换模型: %s -> %s", self.model_name, model_name)
            new_model = YOLO(model_name)
            new_model.to(self.device)
            self.model = new_model
            self.model_name = model_name
            self.class_names = self.model.names
            logger.info("模型切换成功: %s", model_name)
            return True
        except Exception as e:
            logger.error("模型切换失败: %s", e)
            return False
