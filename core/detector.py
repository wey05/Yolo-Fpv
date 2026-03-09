import torch
from ultralytics import YOLO
import cv2
import numpy as np


class PersonDetector:
    DETECT_ALL_CLASSES = True
    
    def __init__(self, model_name='yolov8n.pt'):
        self.model_name = model_name
        self.model = None
        self.device = None
        self.class_names = []
        self._initialize_model()
    
    def _initialize_model(self):
        try:
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"使用GPU加速: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                print("使用CPU模式")
            
            print(f"正在加载模型: {self.model_name}")
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            self.class_names = self.model.names
            print(f"模型加载成功，共{len(self.class_names)}个类别")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def detect(self, frame, conf_threshold=0.5):
        if self.model is None:
            return [], [], [], 0
        
        try:
            results = self.model(frame, conf=conf_threshold, verbose=False, device=self.device)
            
            boxes = []
            confidences = []
            class_ids = []
            person_count = 0
            
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
            print(f"检测错误: {e}")
            return [], [], [], 0
    
    def draw_boxes(self, frame, boxes, confidences, class_ids):
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box
            
            if self.DETECT_ALL_CLASSES:
                np.random.seed(cls_id)
                color = tuple(map(int, np.random.randint(0, 255, 3)))
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
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def switch_model(self, model_name):
        if model_name == self.model_name:
            return False
        try:
            print(f"正在切换模型: {self.model_name} -> {model_name}")
            new_model = YOLO(model_name)
            new_model.to(self.device)
            self.model = new_model
            self.model_name = model_name
            self.class_names = self.model.names
            print(f"模型切换成功: {model_name}")
            return True
        except Exception as e:
            print(f"模型切换失败: {e}")
            return False
