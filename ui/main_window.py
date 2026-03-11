"""
YOLO 实时目标检测系统 - 主窗口
================================
深色现代 UI，支持置信度调节、摄像头选择、分辨率切换。
"""

import logging
import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.camera import RESOLUTION_PRESETS, CameraManager
from ui.theme import (
    BTN_EXIT,
    BTN_REFRESH,
    BTN_SCREENSHOT,
    BTN_START,
    BTN_STOP,
    BTN_SWITCH,
    INFO_LABEL_STYLE,
    SECTION_TITLE_STYLE,
    SLIDER_VALUE_STYLE,
    VIDEO_LABEL_STYLE,
    Colors,
    person_count_style,
)
from utils.thread import DetectionThread

logger = logging.getLogger(__name__)

MODEL_EXTENSIONS = ('.pt', '.pth', '.onnx', '.engine')


class MainWindow(QMainWindow):
    """YOLO 实时目标检测系统主窗口。"""

    MODELS_DIR: str = 'models'

    def __init__(self) -> None:
        super().__init__()
        self.detection_thread: Optional[DetectionThread] = None
        self.current_frame: Optional[np.ndarray] = None
        self.detection_active: bool = False
        self.available_models: list[str] = self._scan_models()
        self.current_model: Optional[str] = (
            self.available_models[0] if self.available_models else None
        )
        self._init_ui()

    # ══════════════════════════════════════════════
    #  模型扫描
    # ══════════════════════════════════════════════

    def _scan_models(self) -> list[str]:
        """扫描 models/ 目录下的所有可用模型文件。"""
        models: list[str] = []
        if os.path.exists(self.MODELS_DIR):
            for f in os.listdir(self.MODELS_DIR):
                if f.endswith(MODEL_EXTENSIONS):
                    full_path = os.path.join(self.MODELS_DIR, f)
                    models.append(full_path)
        if not models:
            default = os.path.join(self.MODELS_DIR, 'yolov8n.pt')
            if os.path.exists(default):
                models.append(default)
            else:
                logger.warning("未找到任何模型文件")
        return sorted(models)

    # ══════════════════════════════════════════════
    #  UI 初始化
    # ══════════════════════════════════════════════

    def _init_ui(self) -> None:
        self.setWindowTitle('YOLO 实时目标检测系统')
        self.setGeometry(100, 100, 1280, 760)
        self.setMinimumSize(1100, 700)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ── 左侧：视频区域 ───────────────────────
        root.addWidget(self._build_video_panel(), stretch=3)

        # ── 右侧：控制面板（使用滚动区域）──────────
        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._build_stats_card())
        right_layout.addWidget(self._build_settings_card())
        right_layout.addWidget(self._build_control_card())
        right_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidget(right_content)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setMinimumWidth(320)
        scroll_area.setMaximumWidth(400)

        root.addWidget(scroll_area, stretch=1)

        self.statusBar().showMessage('就绪 - 点击「开始检测」启动')

    # ── 视频面板 ──────────────────────────────

    def _build_video_panel(self) -> QGroupBox:
        group = QGroupBox("实时视频")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 16, 8, 8)

        self.video_label = QLabel("等待启动摄像头...")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(VIDEO_LABEL_STYLE)
        self.video_label.setFont(QFont('Microsoft YaHei', 14))
        layout.addWidget(self.video_label)

        return group

    # ── 检测统计卡片 ─────────────────────────

    def _build_stats_card(self) -> QGroupBox:
        group = QGroupBox("检测统计")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(4)

        title = QLabel("检测到的人物数量")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Microsoft YaHei', 10, QFont.Bold))
        layout.addWidget(title)

        self.person_count_label = QLabel("0")
        self.person_count_label.setAlignment(Qt.AlignCenter)
        self.person_count_label.setFont(QFont('Consolas', 48, QFont.Bold))
        self.person_count_label.setStyleSheet(
            person_count_style(Colors.STATUS_SAFE)
        )
        self.person_count_label.setMinimumHeight(80)
        layout.addWidget(self.person_count_label)

        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setAlignment(Qt.AlignCenter)
        self.fps_label.setStyleSheet(INFO_LABEL_STYLE)
        self.fps_label.setFont(QFont('Consolas', 10))
        layout.addWidget(self.fps_label)

        self.model_info_label = QLabel("模型: --")
        self.model_info_label.setAlignment(Qt.AlignCenter)
        self.model_info_label.setStyleSheet(INFO_LABEL_STYLE)
        self.model_info_label.setFont(QFont('Microsoft YaHei', 9))
        layout.addWidget(self.model_info_label)

        return group

    # ── 设置卡片（模型 + 置信度 + 摄像头 + 分辨率）─────

    def _build_settings_card(self) -> QGroupBox:
        group = QGroupBox("检测设置")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(4)

        layout.addWidget(self._make_section_title("模型选择"))

        self.model_combo = QComboBox()
        self.model_combo.addItems(self.available_models)
        if self.current_model:
            self.model_combo.setCurrentText(self.current_model)
        layout.addWidget(self.model_combo)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.switch_model_btn = QPushButton("切换模型")
        self.switch_model_btn.setMinimumHeight(32)
        self.switch_model_btn.setStyleSheet(BTN_SWITCH)
        self.switch_model_btn.setEnabled(False)
        self.switch_model_btn.clicked.connect(self._on_switch_model)
        self.switch_model_btn.setToolTip("在检测运行时切换到选定的模型")
        btn_row.addWidget(self.switch_model_btn)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setMinimumHeight(32)
        self.refresh_btn.setStyleSheet(BTN_REFRESH)
        self.refresh_btn.clicked.connect(self._on_refresh_models)
        self.refresh_btn.setToolTip("重新扫描 models/ 目录")
        btn_row.addWidget(self.refresh_btn)

        layout.addLayout(btn_row)

        layout.addWidget(self._make_section_title("置信度阈值"))

        conf_row = QHBoxLayout()
        conf_row.setSpacing(6)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(50)
        self.conf_slider.setTickInterval(5)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        self.conf_slider.setToolTip("调节检测置信度阈值（0.05 ~ 0.95）")
        conf_row.addWidget(self.conf_slider)

        self.conf_value_label = QLabel("0.50")
        self.conf_value_label.setStyleSheet(SLIDER_VALUE_STYLE)
        self.conf_value_label.setAlignment(Qt.AlignCenter)
        self.conf_value_label.setFont(QFont('Consolas', 11, QFont.Bold))
        conf_row.addWidget(self.conf_value_label)

        layout.addLayout(conf_row)

        layout.addWidget(self._make_section_title("摄像头"))

        cam_row = QHBoxLayout()
        cam_row.setSpacing(6)

        self.camera_combo = QComboBox()
        self._populate_cameras()
        cam_row.addWidget(self.camera_combo, stretch=1)

        self.cam_refresh_btn = QPushButton("扫描")
        self.cam_refresh_btn.setMinimumHeight(28)
        self.cam_refresh_btn.setMaximumWidth(55)
        self.cam_refresh_btn.setStyleSheet(BTN_REFRESH)
        self.cam_refresh_btn.clicked.connect(self._on_refresh_cameras)
        self.cam_refresh_btn.setToolTip("重新扫描可用摄像头")
        cam_row.addWidget(self.cam_refresh_btn)

        layout.addLayout(cam_row)

        layout.addWidget(self._make_section_title("分辨率"))

        self.resolution_combo = QComboBox()
        for key in RESOLUTION_PRESETS:
            self.resolution_combo.addItem(key)
        self.resolution_combo.setCurrentText("640x480")
        self.resolution_combo.setToolTip("切换摄像头采集分辨率（需重启检测生效）")
        layout.addWidget(self.resolution_combo)

        return group

    # ── 控制按钮卡片 ─────────────────────────

    def _build_control_card(self) -> QGroupBox:
        group = QGroupBox("控制面板")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(6)

        self.start_btn = QPushButton("开始检测")
        self.start_btn.setMinimumHeight(38)
        self.start_btn.setStyleSheet(BTN_START)
        self.start_btn.clicked.connect(self._on_start)
        self.start_btn.setToolTip("启动摄像头并开始实时检测")
        layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.setMinimumHeight(38)
        self.stop_btn.setStyleSheet(BTN_STOP)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setToolTip("停止检测并释放摄像头")
        layout.addWidget(self.stop_btn)

        self.screenshot_btn = QPushButton("保存截图")
        self.screenshot_btn.setMinimumHeight(38)
        self.screenshot_btn.setStyleSheet(BTN_SCREENSHOT)
        self.screenshot_btn.setEnabled(False)
        self.screenshot_btn.clicked.connect(self._on_screenshot)
        self.screenshot_btn.setToolTip("将当前检测画面保存为图片")
        layout.addWidget(self.screenshot_btn)

        self.exit_btn = QPushButton("退出程序")
        self.exit_btn.setMinimumHeight(38)
        self.exit_btn.setStyleSheet(BTN_EXIT)
        self.exit_btn.clicked.connect(self.close)
        layout.addWidget(self.exit_btn)

        return group

    # ── 辅助方法 ──────────────────────────────

    @staticmethod
    def _make_section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(SECTION_TITLE_STYLE)
        label.setFont(QFont('Microsoft YaHei', 10, QFont.Bold))
        return label

    def _populate_cameras(self) -> None:
        """枚举可用摄像头并填充下拉框。"""
        self.camera_combo.clear()
        cameras = CameraManager.enumerate_cameras(max_test=5)
        if cameras:
            for cam_id in cameras:
                self.camera_combo.addItem(f"摄像头 {cam_id}", cam_id)
        else:
            self.camera_combo.addItem("摄像头 0", 0)

    def _set_detection_ui_state(self, active: bool) -> None:
        """统一切换检测启停时的控件状态。"""
        self.start_btn.setEnabled(not active)
        self.stop_btn.setEnabled(active)
        self.screenshot_btn.setEnabled(active)
        self.switch_model_btn.setEnabled(active)
        # 检测中禁止切换摄像头和分辨率
        self.camera_combo.setEnabled(not active)
        self.resolution_combo.setEnabled(not active)
        self.cam_refresh_btn.setEnabled(not active)

    # ══════════════════════════════════════════════
    #  事件处理
    # ══════════════════════════════════════════════

    def _on_start(self) -> None:
        """开始检测。"""
        if self.detection_active:
            return

        self.current_model = self.model_combo.currentText()
        camera_id: int = self.camera_combo.currentData() or 0
        resolution: str = self.resolution_combo.currentText()
        conf = self.conf_slider.value() / 100.0

        self.detection_thread = DetectionThread(
            camera_id=camera_id,
            model_name=self.current_model,
            resolution=resolution,
        )
        self.detection_thread.set_confidence_threshold(conf)
        self.detection_thread.frame_ready.connect(self._on_frame_ready)
        self.detection_thread.error_occurred.connect(self._on_error)
        self.detection_thread.model_switched.connect(self._on_model_switched)

        self.detection_active = True
        self.detection_thread.start()

        self._set_detection_ui_state(True)
        model_display = os.path.basename(self.current_model).rsplit('.', 1)[0]
        self.model_info_label.setText(f"模型: {model_display}")
        self.statusBar().showMessage('检测中...')
        logger.info("检测已启动 -- 模型: %s, 摄像头: %d, 分辨率: %s",
                     self.current_model, camera_id, resolution)

    def _on_stop(self) -> None:
        """停止检测。"""
        if self.detection_thread and self.detection_active:
            self.detection_thread.stop()
            self.detection_thread = None

        self.detection_active = False
        self.current_frame = None

        self._set_detection_ui_state(False)
        self.video_label.clear()
        self.video_label.setText("检测已停止")
        self.person_count_label.setText("0")
        self.person_count_label.setStyleSheet(
            person_count_style(Colors.STATUS_SAFE)
        )
        self.fps_label.setText("FPS: 0.0")
        self.statusBar().showMessage('已停止')
        logger.info("检测已停止")

    def _on_frame_ready(self, frame: np.ndarray, person_count: int, fps: float) -> None:
        """接收新帧并更新 UI。"""
        self.current_frame = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

        scaled = QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

        # 更新人数
        self.person_count_label.setText(str(person_count))
        color = Colors.STATUS_SAFE if person_count == 0 else Colors.STATUS_WARN
        self.person_count_label.setStyleSheet(person_count_style(color))

        # 更新 FPS
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _on_error(self, msg: str) -> None:
        """处理检测线程错误。"""
        logger.error("检测错误: %s", msg)
        self.statusBar().showMessage(f'错误: {msg}')
        self._on_stop()

    def _on_switch_model(self) -> None:
        """切换模型。"""
        if not (self.detection_thread and self.detection_active):
            return
        new_model = self.model_combo.currentText()
        if new_model != self.current_model:
            self.detection_thread.switch_model(new_model)
            self.statusBar().showMessage(f'正在切换模型到 {os.path.basename(new_model)}...')
        else:
            self.statusBar().showMessage('已是当前模型')

    def _on_model_switched(self, model_name: str) -> None:
        """模型切换完成回调。"""
        self.current_model = model_name
        display = os.path.basename(model_name).rsplit('.', 1)[0]
        self.model_info_label.setText(f"模型: {display}")
        self.statusBar().showMessage(f'已切换到模型: {display}')
        logger.info("模型已切换: %s", model_name)

    def _on_refresh_models(self) -> None:
        """刷新模型列表。"""
        old = set(self.available_models)
        self.available_models = self._scan_models()
        new = set(self.available_models)

        self.model_combo.clear()
        self.model_combo.addItems(self.available_models)

        added = new - old
        removed = old - new
        if added:
            self.statusBar().showMessage(f'发现新模型: {", ".join(os.path.basename(m) for m in added)}')
        elif removed:
            self.statusBar().showMessage(f'模型已移除: {", ".join(os.path.basename(m) for m in removed)}')
        else:
            self.statusBar().showMessage('模型列表已刷新')

        if self.current_model in self.available_models:
            self.model_combo.setCurrentText(self.current_model)

    def _on_refresh_cameras(self) -> None:
        """刷新摄像头列表。"""
        self._populate_cameras()
        self.statusBar().showMessage('摄像头列表已刷新')

    def _on_conf_changed(self, value: int) -> None:
        """置信度滑块值变化。"""
        threshold = value / 100.0
        self.conf_value_label.setText(f"{threshold:.2f}")
        if self.detection_thread and self.detection_active:
            self.detection_thread.set_confidence_threshold(threshold)

    def _on_screenshot(self) -> None:
        """保存当前帧截图。"""
        if self.current_frame is None:
            self.statusBar().showMessage('无可用截图')
            return

        screenshot_dir = 'screenshots'
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(screenshot_dir, f"screenshot_{timestamp}.jpg")
        cv2.imwrite(filepath, self.current_frame)
        self.statusBar().showMessage(f'截图已保存: {filepath}')
        logger.info("截图已保存: %s", filepath)

    # ══════════════════════════════════════════════
    #  窗口事件
    # ══════════════════════════════════════════════

    def closeEvent(self, event) -> None:  # noqa: N802
        """窗口关闭时确保线程安全退出。"""
        if self.detection_thread and self.detection_active:
            self.detection_thread.stop()
        event.accept()
