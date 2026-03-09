import time
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from core.detector import PersonDetector
from core.camera import CameraManager
import cv2
import numpy as np


class DetectionThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, int, float)
    error_occurred = pyqtSignal(str)
    model_switched = pyqtSignal(str)
    
    def __init__(self, camera_id=0, model_name='yolov8n.pt'):
        super().__init__()
        self.camera_id = camera_id
        self.model_name = model_name
        self.running = False
        self.detector = None
        self.camera = None
        self.conf_threshold = 0.5
        self.pending_model = None
    
    def run(self):
        try:
            self.detector = PersonDetector(self.model_name)
            self.camera = CameraManager(self.camera_id)
            
            if not self.camera.open():
                self.error_occurred.emit("无法打开摄像头")
                return
            
            self.running = True
            fps_counter = 0
            fps_timer = time.time()
            fps = 0.0
            
            while self.running:
                if self.pending_model:
                    if self.detector.switch_model(self.pending_model):
                        self.model_name = self.pending_model
                        self.model_switched.emit(self.pending_model)
                    self.pending_model = None
                
                frame = self.camera.read_frame()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                start_time = time.time()
                
                boxes, confidences, class_ids, person_count = self.detector.detect(
                    frame, self.conf_threshold
                )
                
                frame_with_boxes = self.detector.draw_boxes(frame, boxes, confidences, class_ids)
                
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    fps = fps_counter / (time.time() - fps_timer)
                    fps_counter = 0
                    fps_timer = time.time()
                
                self.frame_ready.emit(frame_with_boxes, person_count, fps)
                
                elapsed = time.time() - start_time
                sleep_time = max(0, (1.0 / 30.0) - elapsed)
                time.sleep(sleep_time)
                
        except Exception as e:
            self.error_occurred.emit(f"检测线程错误: {str(e)}")
        finally:
            if self.camera:
                self.camera.close()
    
    def stop(self):
        self.running = False
        self.wait()
    
    def set_confidence_threshold(self, threshold):
        self.conf_threshold = threshold
    
    def switch_model(self, model_name):
        self.pending_model = model_name
