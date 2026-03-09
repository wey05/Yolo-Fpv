# YOLO实时人员检测系统

基于YOLOv8和PyQt5的实时摄像头人员检测应用。

## 功能特性

- ✅ 实时摄像头人员检测
- ✅ 现代化PyQt5界面
- ✅ GPU/CPU自动切换
- ✅ 实时显示人物数量
- ✅ FPS监控
- ✅ 检测框可视化（颜色区分置信度）
- ✅ 一键截图保存

## 项目结构

```
yolo/
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖管理
├── ui/
│   ├── __init__.py
│   └── main_window.py     # PyQt5主窗口
├── core/
│   ├── __init__.py
│   ├── camera.py          # 摄像头管理
│   └── detector.py        # YOLOv8检测器
└── utils/
    ├── __init__.py
    └── thread.py          # 工作线程
```

## 安装步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行程序

```bash
python main.py
```

首次运行会自动下载YOLOv8n模型（约6MB）。

## 使用说明

1. **开始检测**: 点击"开始检测"按钮启动摄像头
2. **查看结果**: 实时视频显示检测到的人物（绿色框=高置信度，黄色框=中置信度）
3. **保存截图**: 点击"保存截图"保存当前画面
4. **停止检测**: 点击"停止检测"关闭摄像头

## 性能参数

- **模型**: YOLOv8n (nano版本)
- **CPU模式**: 15-20 FPS
- **GPU模式**: 30+ FPS
- **检测精度**: Person类别 mAP>90%

## 系统要求

- Python 3.8+
- 摄像头设备
- (可选) NVIDIA GPU + CUDA

## 故障排除

### 摄像头无法打开
- 检查摄像头是否被其他程序占用
- 尝试修改`camera_id`参数（0, 1, 2...）

### 检测速度慢
- 确保已安装GPU版本的PyTorch
- 或尝试更小的输入分辨率

### 模型下载失败
- 手动下载: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
- 放置在项目根目录

## 技术栈

- **YOLOv8**: 目标检测模型
- **PyQt5**: GUI框架
- **OpenCV**: 图像处理
- **PyTorch**: 深度学习框架

## 许可证

MIT License
