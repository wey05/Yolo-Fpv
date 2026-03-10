"""
YOLO 实时目标检测系统 - 程序入口
==================================
"""

import logging
import sys

from PyQt5.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import GLOBAL_STYLESHEET


def _setup_logging() -> None:
    """配置全局日志格式。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    _setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("YOLO 实时目标检测系统启动中...")

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(GLOBAL_STYLESHEET)

    window = MainWindow()
    window.show()

    logger.info("窗口已显示，进入事件循环")
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
