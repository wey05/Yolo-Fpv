# 添加快捷键N切换显示模式功能计划

## 功能概述
添加快捷键N来切换窗口显示模式：
- **正常模式**：显示视频区域和右侧控制面板（当前默认模式）
- **全屏视频模式**：只显示实时视频界面，隐藏右侧控制面板

## 实现步骤

### 步骤1：添加状态变量
在`MainWindow`类的`__init__`方法中添加状态变量：
- `self.is_fullscreen_mode: bool = False` - 标记当前是否为全屏视频模式
- `self.right_panel: Optional[QScrollArea] = None` - 保存对右侧控制面板的引用

### 步骤2：保存右侧面板引用
在`_init_ui`方法中，将创建的`scroll_area`保存到`self.right_panel`：
```python
self.right_panel = scroll_area
root.addWidget(scroll_area, stretch=1)
```

### 步骤3：实现快捷键处理
在`MainWindow`类中添加`keyPressEvent`方法：
```python
def keyPressEvent(self, event) -> None:
    """处理键盘事件。"""
    if event.key() == Qt.Key_N:
        self._toggle_display_mode()
    super().keyPressEvent(event)
```

### 步骤4：实现显示模式切换逻辑
在`MainWindow`类中添加`_toggle_display_mode`方法：
```python
def _toggle_display_mode(self) -> None:
    """切换显示模式（正常模式/全屏视频模式）。"""
    self.is_fullscreen_mode = not self.is_fullscreen_mode
    
    if self.is_fullscreen_mode:
        # 切换到全屏视频模式：隐藏右侧面板
        self.right_panel.setVisible(False)
        self.statusBar().showMessage('全屏视频模式 - 按N键返回正常模式')
    else:
        # 切换到正常模式：显示右侧面板
        self.right_panel.setVisible(True)
        self.statusBar().showMessage('正常模式 - 按N键切换到全屏视频模式')
```

### 步骤5：更新README文档
在README.md的更新日志中添加新功能说明：
```markdown
### 2026-03-11 - 快捷键切换显示模式
- **添加快捷键N切换显示模式**
  - 正常模式：显示视频区域和右侧控制面板
  - 全屏视频模式：只显示实时视频界面
  - 按N键在两种模式之间切换
  - 状态栏显示当前模式提示
  - 修改文件：`ui/main_window.py`
```

## 技术细节

### 状态管理
- 使用布尔变量`is_fullscreen_mode`跟踪当前模式
- 模式切换时更新状态栏提示信息

### UI布局
- 正常模式：视频区域(stretch=3) + 控制面板(stretch=1)
- 全屏视频模式：视频区域(stretch=3)，控制面板隐藏

### 用户体验
- 快捷键N快速切换模式
- 状态栏实时显示当前模式
- 切换时保持窗口大小不变

## 修改文件列表
- `ui/main_window.py` - 添加快捷键处理和模式切换逻辑
- `README.md` - 更新功能说明

## 测试要点
1. 按N键能够正常切换显示模式
2. 正常模式下右侧控制面板可见
3. 全屏视频模式下右侧控制面板隐藏
4. 状态栏正确显示当前模式
5. 切换模式不影响视频播放和检测功能
