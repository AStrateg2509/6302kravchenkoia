"""Конфигурация логгера metetl: подробный файл-лог + краткая консоль."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

LOGGER_NAME = "metetl"
DEFAULT_LOG_FILE = Path("logs") / "metetl.log"

_FILE_FORMAT = "%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
_CONSOLE_FORMAT = "%(levelname)s %(message)s"

_configured = False


def configure(log_file: Optional[Path] = None, console_level: int = logging.INFO) -> logging.Logger:
    """Создаёт логгер metetl. Идемпотентна — повторные вызовы не дублируют хэндлеры."""
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    if _configured:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    target = Path(log_file) if log_file is not None else DEFAULT_LOG_FILE
    target.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    _configured = True
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Возвращает дочерний логгер metetl.<name> или сам корень."""
    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
