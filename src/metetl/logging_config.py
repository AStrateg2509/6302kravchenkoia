"""Конфигурация логгера metetl: подробный файл-лог + краткая консоль."""

from __future__ import annotations

import json
import logging
import logging.config
from pathlib import Path
from typing import Optional

LOGGER_NAME = "metetl"

_CFG_PATH = Path(__file__).with_name("logging_config.json")

_configured = False


def configure(log_file: Optional[Path] = None, console_level: int = logging.INFO) -> logging.Logger:
    """Читает JSON, превращает в dict и настраивает логгер через dictConfig."""
    global _configured
    if _configured:
        return logging.getLogger(LOGGER_NAME)

    with open(_CFG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    if log_file is not None:
        cfg["handlers"]["file"]["filename"] = str(log_file)

    cfg["handlers"]["console"]["level"] = logging.getLevelName(console_level)

    log_path = Path(cfg["handlers"]["file"]["filename"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(cfg)

    _configured = True
    return logging.getLogger(LOGGER_NAME)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Возвращает дочерний логгер metetl.<name> или сам корень."""
    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
