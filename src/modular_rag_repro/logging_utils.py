"""复现版项目的日志工具。"""

from __future__ import annotations

import logging
from typing import Optional


_LOGGING_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """配置全局日志。

    只在首次调用时真正生效，避免脚本多次导入时重复初始化。
    当前阶段希望 `--verbose` 主要看到我们自己的调试信息，
    所以会主动压低第三方依赖的日志级别，避免终端被底层 HTTP / DB 日志刷屏。
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    resolved_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # 只保留项目自身的详细日志，第三方依赖默认压到 WARNING。
    logging.getLogger("modular_rag_repro").setLevel(resolved_level)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("posthog").setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """返回一个 logger 实例。"""
    return logging.getLogger(name)
