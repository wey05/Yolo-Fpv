"""
YOLO 实时目标检测系统 - 统一深色主题系统
=============================================
集中管理所有 QSS 样式，消除内联样式重复。
"""


# ── 配色方案 ──────────────────────────────────

class Colors:
    """深色主题配色常量。"""

    # 基础背景
    BG_PRIMARY = "#1a1b2e"       # 主窗口背景（深蓝黑）
    BG_SECONDARY = "#232438"     # 卡片/面板背景
    BG_TERTIARY = "#2c2d44"      # 输入框 / 次级元素背景
    BG_VIDEO = "#12131f"         # 视频区域背景（最深）

    # 边框
    BORDER = "#3a3b55"           # 普通边框
    BORDER_LIGHT = "#4a4b65"     # 高亮边框
    BORDER_FOCUS = "#7c3aed"     # 聚焦边框（紫色）

    # 文字
    TEXT_PRIMARY = "#e8e9f0"     # 主文字
    TEXT_SECONDARY = "#9a9bb5"   # 次要文字
    TEXT_MUTED = "#6b6c85"       # 淡化文字

    # 强调色
    ACCENT = "#7c3aed"           # 主强调色（紫色）
    ACCENT_HOVER = "#6d28d9"     # 强调色 hover
    ACCENT_PRESSED = "#5b21b6"   # 强调色 pressed

    # 功能色
    GREEN = "#10b981"            # 成功 / 开始
    GREEN_HOVER = "#059669"
    GREEN_PRESSED = "#047857"

    RED = "#ef4444"              # 危险 / 停止
    RED_HOVER = "#dc2626"
    RED_PRESSED = "#b91c1c"

    BLUE = "#3b82f6"             # 信息 / 截图
    BLUE_HOVER = "#2563eb"
    BLUE_PRESSED = "#1d4ed8"

    ORANGE = "#f59e0b"           # 警告 / 退出
    ORANGE_HOVER = "#d97706"
    ORANGE_PRESSED = "#b45309"

    SLATE = "#64748b"            # 中性 / 刷新
    SLATE_HOVER = "#475569"
    SLATE_PRESSED = "#334155"

    # 状态色
    STATUS_SAFE = "#10b981"      # 检测到 0 人
    STATUS_WARN = "#fbbf24"      # 检测到人

    # 禁用
    DISABLED_BG = "#3a3b55"
    DISABLED_TEXT = "#6b6c85"


# ── 按钮样式工厂 ─────────────────────────────

def button_style(
    bg: str,
    hover: str,
    pressed: str,
    *,
    text: str = Colors.TEXT_PRIMARY,
    disabled_bg: str = Colors.DISABLED_BG,
    disabled_text: str = Colors.DISABLED_TEXT,
    radius: int = 8,
    font_size: int = 14,
) -> str:
    """生成统一的 QPushButton QSS 样式。"""
    return f"""
        QPushButton {{
            background-color: {bg};
            color: {text};
            border: none;
            border-radius: {radius}px;
            font-size: {font_size}px;
            font-weight: bold;
            padding: 8px 16px;
        }}
        QPushButton:hover {{
            background-color: {hover};
        }}
        QPushButton:pressed {{
            background-color: {pressed};
        }}
        QPushButton:disabled {{
            background-color: {disabled_bg};
            color: {disabled_text};
        }}
    """


# ── 全局样式表 ───────────────────────────────

GLOBAL_STYLESHEET = f"""
    /* ─ 主窗口 ─ */
    QMainWindow {{
        background-color: {Colors.BG_PRIMARY};
    }}

    /* ─ 分组框 ─ */
    QGroupBox {{
        font-weight: bold;
        font-size: 13px;
        color: {Colors.TEXT_SECONDARY};
        border: 1px solid {Colors.BORDER};
        border-radius: 10px;
        margin-top: 14px;
        padding: 18px 12px 12px 12px;
        background-color: {Colors.BG_SECONDARY};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 14px;
        padding: 0 8px;
        color: {Colors.TEXT_PRIMARY};
    }}

    /* ─ 标签 ─ */
    QLabel {{
        color: {Colors.TEXT_PRIMARY};
        background: transparent;
    }}

    /* ─ 下拉框 ─ */
    QComboBox {{
        background-color: {Colors.BG_TERTIARY};
        color: {Colors.TEXT_PRIMARY};
        border: 1px solid {Colors.BORDER};
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 13px;
        min-height: 28px;
    }}
    QComboBox:hover {{
        border: 1px solid {Colors.ACCENT};
    }}
    QComboBox:focus {{
        border: 1px solid {Colors.ACCENT};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 30px;
    }}
    QComboBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 7px solid {Colors.TEXT_SECONDARY};
        margin-right: 10px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {Colors.BG_TERTIARY};
        color: {Colors.TEXT_PRIMARY};
        selection-background-color: {Colors.ACCENT};
        selection-color: white;
        border: 1px solid {Colors.BORDER};
        border-radius: 4px;
        padding: 4px;
        outline: none;
    }}

    /* ─ 滑块 ─ */
    QSlider::groove:horizontal {{
        border: none;
        height: 6px;
        background-color: {Colors.BG_TERTIARY};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background-color: {Colors.ACCENT};
        border: 2px solid {Colors.BG_SECONDARY};
        width: 18px;
        height: 18px;
        margin: -7px 0;
        border-radius: 10px;
    }}
    QSlider::handle:horizontal:hover {{
        background-color: {Colors.ACCENT_HOVER};
    }}
    QSlider::sub-page:horizontal {{
        background-color: {Colors.ACCENT};
        border-radius: 3px;
    }}

    /* ─ 状态栏 ─ */
    QStatusBar {{
        background-color: {Colors.BG_SECONDARY};
        color: {Colors.TEXT_SECONDARY};
        border-top: 1px solid {Colors.BORDER};
        font-size: 12px;
        padding: 4px 8px;
    }}

    /* ─ 滚动区域 ─ */
    QScrollArea {{
        background-color: {Colors.BG_PRIMARY};
        border: none;
        border-radius: 10px;
    }}
    QScrollArea > QWidget > QWidget {{
        background-color: transparent;
    }}

    /* ─ 滚动条 ─ */
    QScrollBar:vertical {{
        background: {Colors.BG_SECONDARY};
        width: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background: {Colors.BORDER};
        border-radius: 4px;
        min-height: 20px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    /* ─ 工具提示 ─ */
    QToolTip {{
        background-color: {Colors.BG_TERTIARY};
        color: {Colors.TEXT_PRIMARY};
        border: 1px solid {Colors.BORDER};
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 12px;
    }}
"""


# ── 特殊组件样式 ─────────────────────────────

VIDEO_LABEL_STYLE = f"""
    QLabel {{
        background-color: {Colors.BG_VIDEO};
        border: 2px solid {Colors.BORDER};
        border-radius: 10px;
        color: {Colors.TEXT_MUTED};
        font-size: 16px;
    }}
"""


def person_count_style(color: str) -> str:
    """根据检测状态生成人数显示标签的样式。"""
    return f"""
        QLabel {{
            color: {color};
            background-color: {Colors.BG_VIDEO};
            border: 2px solid {color};
            border-radius: 12px;
            padding: 16px;
        }}
    """


INFO_LABEL_STYLE = f"""
    QLabel {{
        color: {Colors.TEXT_SECONDARY};
        font-size: 12px;
        padding: 2px 0;
    }}
"""

SECTION_TITLE_STYLE = f"""
    QLabel {{
        color: {Colors.TEXT_PRIMARY};
        font-size: 12px;
        font-weight: bold;
        padding: 4px 0 2px 0;
    }}
"""

SLIDER_VALUE_STYLE = f"""
    QLabel {{
        color: {Colors.ACCENT};
        font-size: 13px;
        font-weight: bold;
        min-width: 36px;
    }}
"""


# ── 预定义按钮样式 ───────────────────────────

BTN_START = button_style(Colors.GREEN, Colors.GREEN_HOVER, Colors.GREEN_PRESSED)
BTN_STOP = button_style(Colors.RED, Colors.RED_HOVER, Colors.RED_PRESSED)
BTN_SCREENSHOT = button_style(Colors.BLUE, Colors.BLUE_HOVER, Colors.BLUE_PRESSED)
BTN_EXIT = button_style(Colors.ORANGE, Colors.ORANGE_HOVER, Colors.ORANGE_PRESSED, text="#1a1b2e")
BTN_SWITCH = button_style(Colors.ACCENT, Colors.ACCENT_HOVER, Colors.ACCENT_PRESSED)
BTN_REFRESH = button_style(Colors.SLATE, Colors.SLATE_HOVER, Colors.SLATE_PRESSED)
