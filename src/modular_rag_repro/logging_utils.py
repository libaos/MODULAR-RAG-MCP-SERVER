"""复现版项目的日志工具。"""

from __future__ import annotations

import logging
from typing import Optional


_LOGGING_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """配置全局日志。

    只在首次调用时真正生效，避免脚本多次导入时重复初始化。
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _LOGGING_CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """返回一个 logger 实例。"""
    return logging.getLogger(name)
