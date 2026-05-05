"""Декораторы общего назначения."""

from __future__ import annotations

import asyncio
import time
from functools import wraps
from typing import Any, Callable

from metetl.logging_config import get_logger


def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    """Замеряет и логирует время выполнения. Поддерживает sync и async функции."""
    log = get_logger("timing")

    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                log.debug("%s выполнена за %.4f с", func.__name__, time.perf_counter() - start)

        return async_wrapper

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            log.debug("%s выполнена за %.4f с", func.__name__, time.perf_counter() - start)

    return wrapper
