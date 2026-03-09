import cv2


class CameraManager:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.is_opened = False
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30
    
    def open(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                print(f"无法打开摄像头 {self.camera_id}")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.is_opened = True
            print(f"摄像头已打开: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"摄像头初始化错误: {e}")
            return False
    
    def read_frame(self):
        if not self.is_opened or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            print("读取帧失败")
            return None
        
        return frame
    
    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False
        print("摄像头已关闭")
    
    def get_camera_info(self):
        return {
            'camera_id': self.camera_id,
            'width': self.frame_width,
            'height': self.frame_height,
            'fps': self.fps,
            'is_opened': self.is_opened
        }
