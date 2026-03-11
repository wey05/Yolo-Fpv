"""
YOLO 实时目标检测系统 - 配置管理
==================================
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """应用配置类。"""

    last_model: str = ""
    camera_id: int = 0
    resolution: str = "640x480"
    confidence_threshold: float = 0.5
    window_x: int = 100
    window_y: int = 100
    window_width: int = 1280
    window_height: int = 760

    def to_dict(self) -> dict:
        """转换为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'AppConfig':
        """从字典创建配置。"""
        return cls(**data)


class ConfigManager:
    """配置管理器，负责配置的加载和保存。"""

    CONFIG_FILE = "config.json"

    def __init__(self, config_dir: Optional[str] = None) -> None:
        if config_dir:
            self.config_path = Path(config_dir) / self.CONFIG_FILE
        else:
            self.config_path = Path(self.CONFIG_FILE)
        self.config: AppConfig = AppConfig()

    def load(self) -> AppConfig:
        """加载配置文件。"""
        if not self.config_path.exists():
            logger.info("配置文件不存在，使用默认配置")
            return self.config

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.config = AppConfig.from_dict(data)
            logger.info("配置加载成功: %s", self.config_path)
            return self.config
        except Exception as e:
            logger.error("配置加载失败: %s", e)
            return self.config

    def save(self) -> bool:
        """保存配置到文件。"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=4, ensure_ascii=False)
            logger.info("配置保存成功: %s", self.config_path)
            return True
        except Exception as e:
            logger.error("配置保存失败: %s", e)
            return False

    def update(self, **kwargs) -> None:
        """更新配置。"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning("未知配置项: %s", key)

    def get(self, key: str, default=None):
        """获取配置项。"""
        return getattr(self.config, key, default)
