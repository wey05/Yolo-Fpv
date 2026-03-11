# YOLO实时目标检测系统

基于YOLO和PyQt5的实时摄像头目标检测应用，支持多种YOLO模型动态切换，现代化深色UI设计。

## 功能特性

- ✅ 实时摄像头目标检测（支持80+类别）
- ✅ 多模型支持（YOLOv8n/v10n/YOLO26n/m/x）
- ✅ 运行时动态切换模型
- ✅ **置信度阈值实时调节**（0.05 ~ 0.95）
- ✅ **摄像头设备选择**
- ✅ **分辨率切换**（640x480 / 1280x720 / 1920x1080）
- ✅ 现代化深色主题PyQt5界面
- ✅ GPU/CPU自动切换
- ✅ 实时显示人物数量
- ✅ FPS监控
- ✅ 检测框可视化（类别颜色区分）
- ✅ 一键截图保存
- ✅ Git LFS支持大模型文件管理

## 项目结构

```
yolo/
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖管理
├── models/                 # 模型文件目录
│   ├── yolov8n.pt         # YOLOv8 nano (6.3MB)
│   ├── yolov10n.pt        # YOLOv10 nano (5.6MB)
│   ├── yolo26n.pt         # YOLO26 nano (5.3MB)
│   ├── yolo26m.pt         # YOLO26 medium (43MB)
│   └── yolo26x.pt         # YOLO26 extra large (114MB)
├── ui/
│   ├── __init__.py
│   ├── theme.py           # 统一主题系统（配色、样式）
│   └── main_window.py     # PyQt5主窗口
├── core/
│   ├── __init__.py
│   ├── detector.py        # YOLO检测器（ObjectDetector）
│   └── camera.py          # 摄像头管理器
└── utils/
    ├── __init__.py
    └── thread.py          # 检测工作线程（线程安全）
```

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/wey05/Yolo-Fpv.git
cd Yolo-Fpv
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装Git LFS（用于下载模型）

```bash
git lfs install
git lfs pull
```

### 4. 运行程序

```bash
python main.py
```

## 使用说明

1. **选择模型**: 在"检测设置"区域的下拉框中选择要使用的YOLO模型
2. **调节置信度**: 拖动"置信度阈值"滑块调整检测灵敏度（默认0.50）
3. **选择摄像头**: 下拉选择摄像头设备，点击"扫描"刷新设备列表
4. **选择分辨率**: 选择摄像头采集分辨率（需重启检测生效）
5. **开始检测**: 点击"开始检测"按钮启动摄像头
6. **查看结果**: 实时视频显示检测到的物体，右侧显示人物数量和FPS
7. **切换模型**: 检测过程中可随时切换模型（点击"切换模型"按钮）
8. **保存截图**: 点击"保存截图"保存当前画面到 `screenshots/` 目录
9. **停止检测**: 点击"停止检测"关闭摄像头

## 支持的模型

| 模型     | 大小  | 速度 | 精度 | 适用场景           |
| -------- | ----- | ---- | ---- | ------------------ |
| YOLOv8n  | 6.3MB | 最快 | 较低 | 实时检测、低端设备 |
| YOLOv10n | 5.6MB | 最快 | 较低 | 实时检测、资源受限 |
| YOLO26n  | 5.3MB | 最快 | 中等 | 平衡速度与精度     |
| YOLO26m  | 43MB  | 中等 | 较高 | 精度优先场景       |
| YOLO26x  | 114MB | 较慢 | 最高 | 最高精度要求       |

## 性能参数

- **CPU模式**: 15-20 FPS（YOLOv8n）
- **GPU模式**: 30-60 FPS（YOLOv8n，取决于GPU性能）
- **检测类别**: 80+ 类别（COCO数据集）
- **检测精度**: mAP@0.5 > 50%（YOLOv8n）

## 系统要求

- Python 3.8+
- 摄像头设备
- （可选）NVIDIA GPU + CUDA

## 故障排除

### 摄像头无法打开

- 检查摄像头是否被其他程序占用
- 尝试在摄像头下拉框中选择其他设备（摄像头 0/1/2...）

### 检测速度慢

- 确保已安装GPU版本的PyTorch
- 使用更小的模型（如YOLOv8n或YOLO26n）
- 降低分辨率设置

### 模型文件缺失

```bash
# 确保已安装Git LFS
git lfs install
# 拉取模型文件
git lfs pull
```

### GPU未被识别

```bash
# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"
# 安装GPU版本的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 技术栈

- **YOLO**: 目标检测模型（Ultralytics）
- **PyQt5**: GUI框架
- **OpenCV**: 图像处理
- **PyTorch**: 深度学习框架
- **Git LFS**: 大文件管理

## 开发说明

### 检测所有类别

默认检测所有80+类别。如需仅检测人员，修改 `core/detector.py`:

```python
DETECT_ALL_CLASSES = False  # 改为False仅检测人员
```

### 添加新模型

将新的 `.pt` 模型文件放入 `models/` 目录，点击界面上的"刷新"按钮即可识别。

### 主题定制

所有UI样式集中在 `ui/theme.py`，可修改 `Colors` 类调整配色方案。

### 代码规范

- 全部函数添加了类型注解（Type Hints）
- 使用 `logging` 模块替代 `print` 输出日志
- 检测线程使用 `threading.Lock` 保证线程安全

## 更新日志

### 2026-03-11 - 截图保存动画效果补丁

### 2026-03-11 - 截图保存动画效果

- **添加截图保存闪光动画**
  - 截图保存时显示白色闪光效果
  - 模拟相机快门效果，提升用户体验
  - 动画持续约0.5秒后自动消失
  - 修改文件：`ui/main_window.py`

### 2026-03-11 - 配置管理与用户体验增强

- **添加配置文件持久化功能**

  - 支持保存和加载用户配置（JSON格式）
  - 自动保存最后使用的模型、摄像头ID、分辨率、置信度阈值
  - 自动保存窗口位置和大小，下次启动恢复
  - 修改文件：`core/config.py`, `ui/main_window.py`
- **添加摄像头打开进度指示器**

  - 显示摄像头打开进度（10% → 30% → 50% → 100%）
  - 实时显示状态消息（"正在初始化检测器..."、"正在打开摄像头..."等）
  - 进度完成后自动隐藏
  - 修改文件：`utils/thread.py`, `ui/main_window.py`
- **添加模型切换加载动画**

  - 显示模型切换进度和状态
  - 添加旋转加载动画（4个渐变圆点）
  - 提升用户体验，减少等待焦虑
  - 修改文件：`utils/thread.py`, `ui/main_window.py`, `ui/theme.py`

### 2026-03-11 - UI界面优化

- **优化PyQt5界面布局，解决按钮重叠问题**
  - 添加QScrollArea滚动区域支持，防止窗口缩小时按钮重叠
  - 优化最小窗口尺寸（1100x700），确保布局合理
  - 优化控件间距和尺寸，使界面更紧凑美观
  - 修复右侧操作栏背景色，与深色主题保持一致
  - 修改文件：`ui/main_window.py`, `ui/theme.py`

### 2026-03-11 - 摄像头兼容性修复

- **修复Windows系统摄像头1无法打开的问题**
  - 在 `CameraManager`中使用 `cv2.CAP_DSHOW`后端替代默认后端
  - 提高了多摄像头场景下的兼容性和稳定性
  - 修改文件：`core/camera.py`
  - 测试结果：摄像头0和摄像头1均可正常打开和读取

## 许可证

MIT License
