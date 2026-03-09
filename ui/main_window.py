from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QGroupBox, QApplication, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
import sys
import os
import cv2
import numpy as np
from datetime import datetime
from utils.thread import DetectionThread


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detection_thread = None
        self.current_frame = None
        self.detection_active = False
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle('YOLO实时人员检测系统')
        self.setGeometry(100, 100, 1200, 700)
        self.setMinimumSize(800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        video_group = QGroupBox("实时视频")
        video_layout = QVBoxLayout(video_group)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 2px solid #555;
                border-radius: 5px;
                color: #888;
                font-size: 16px;
            }
        """)
        self.video_label.setText("等待启动摄像头...")
        video_layout.addWidget(self.video_label)
        
        main_layout.addWidget(video_group, stretch=3)
        
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        stats_group = QGroupBox("检测统计")
        stats_layout = QVBoxLayout(stats_group)
        
        person_count_label_title = QLabel("检测到的人物数量")
        person_count_label_title.setAlignment(Qt.AlignCenter)
        person_count_label_title.setFont(QFont('Arial', 12, QFont.Bold))
        stats_layout.addWidget(person_count_label_title)
        
        self.person_count_label = QLabel("0")
        self.person_count_label.setAlignment(Qt.AlignCenter)
        self.person_count_label.setFont(QFont('Arial', 48, QFont.Bold))
        self.person_count_label.setStyleSheet("""
            QLabel {
                color: #00ff00;
                background-color: #1a1a1a;
                border: 3px solid #00ff00;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        stats_layout.addWidget(self.person_count_label)
        
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setAlignment(Qt.AlignCenter)
        self.fps_label.setFont(QFont('Arial', 10))
        stats_layout.addWidget(self.fps_label)
        
        self.model_label = QLabel("模型: YOLOv8n")
        self.model_label.setAlignment(Qt.AlignCenter)
        self.model_label.setFont(QFont('Arial', 9))
        stats_layout.addWidget(self.model_label)
        
        control_layout.addWidget(stats_group)
        
        control_group = QGroupBox("控制面板")
        control_group_layout = QVBoxLayout(control_group)
        
        self.start_btn = QPushButton("开始检测")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        control_group_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #666;
            }
        """)
        control_group_layout.addWidget(self.stop_btn)
        
        self.screenshot_btn = QPushButton("保存截图")
        self.screenshot_btn.setMinimumHeight(40)
        self.screenshot_btn.clicked.connect(self.save_screenshot)
        self.screenshot_btn.setEnabled(False)
        self.screenshot_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #666;
            }
        """)
        control_group_layout.addWidget(self.screenshot_btn)
        
        self.exit_btn = QPushButton("退出程序")
        self.exit_btn.setMinimumHeight(40)
        self.exit_btn.clicked.connect(self.close)
        self.exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e68900;
            }
        """)
        control_group_layout.addWidget(self.exit_btn)
        
        control_layout.addWidget(control_group)
        
        control_layout.addStretch()
        
        main_layout.addWidget(control_panel, stretch=1)
        
        self.statusBar().showMessage('就绪 - 点击"开始检测"启动')
    
    def start_detection(self):
        if self.detection_active:
            return
        
        self.detection_thread = DetectionThread(camera_id=0, model_name='yolov8n.pt')
        self.detection_thread.frame_ready.connect(self.update_frame)
        self.detection_thread.error_occurred.connect(self.handle_error)
        
        self.detection_active = True
        self.detection_thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.screenshot_btn.setEnabled(True)
        self.statusBar().showMessage('检测中...')
    
    def stop_detection(self):
        if self.detection_thread and self.detection_active:
            self.detection_thread.stop()
            self.detection_thread = None
        
        self.detection_active = False
        self.current_frame = None
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.screenshot_btn.setEnabled(False)
        
        self.video_label.setText("检测已停止")
        self.person_count_label.setText("0")
        self.fps_label.setText("FPS: 0.0")
        self.statusBar().showMessage('已停止')
    
    def update_frame(self, frame, person_count, fps):
        self.current_frame = frame.copy()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
        
        self.person_count_label.setText(str(person_count))
        self.fps_label.setText(f"FPS: {fps:.1f}")
        
        color = "#00ff00" if person_count == 0 else "#ffff00"
        self.person_count_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                background-color: #1a1a1a;
                border: 3px solid {color};
                border-radius: 10px;
                padding: 20px;
                font-size: 48px;
                font-weight: bold;
            }}
        """)
    
    def handle_error(self, error_msg):
        self.statusBar().showMessage(f'错误: {error_msg}')
        self.stop_detection()
    
    def save_screenshot(self):
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            
            if not os.path.exists('screenshots'):
                os.makedirs('screenshots')
            
            filepath = os.path.join('screenshots', filename)
            cv2.imwrite(filepath, self.current_frame)
            self.statusBar().showMessage(f'截图已保存: {filepath}')
        else:
            self.statusBar().showMessage('无可用截图')
    
    def closeEvent(self, event):
        if self.detection_thread and self.detection_active:
            self.detection_thread.stop()
        event.accept()
