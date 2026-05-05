"""Конвейер скачивания и обработки изображений (Lab 4 + impl-dispatch для Lab 5)."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, List, Optional, Tuple

import aiofiles
import aiohttp
import cv2
import numpy as np

from metetl.decorators import timeit
from metetl.images.models import Artwork, ColorArtwork, GrayscaleArtwork
from metetl.logging_config import get_logger

_log = get_logger("images.processing")

ImplName = str  # "numpy" | "opencv" | "c"
_VALID_IMPLS: Tuple[ImplName, ...] = ("numpy", "opencv", "c")

_MET_API = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"

DownloadPayload = Tuple[int, str, bytes]
ConvolveJob = Tuple[int, str, bytes, ImplName]


def _gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


def _convolve_numpy(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Чистый numpy через метод apply_filter моделей."""
    if img.ndim == 3 and img.shape[2] == 3:
        artwork: Artwork = ColorArtwork(img, {})
    else:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        artwork = GrayscaleArtwork(img, {})
    return artwork.apply_filter(kernel).image


def _convolve_opencv(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img, -1, kernel)


def _convolve_c(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Через C-расширение. Возвращает uint8 такой же размерности, как вход."""
    from metetl.images import c_ext

    if not c_ext.is_available():
        raise RuntimeError("C-расширение недоступно: переустановите пакет с собранным image_ops")
    out = c_ext.convolve_uint8(img, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


_DISPATCH = {
    "numpy": _convolve_numpy,
    "opencv": _convolve_opencv,
    "c": _convolve_c,
}


def _convolve_in_process(payload: ConvolveJob) -> DownloadPayload:
    """
    ProcessPoolExecutor-воркер: декодирует PNG, применяет свёртку выбранной
    реализацией, возвращает PNG-байты результата. Логгер настраивается
    лениво в каждом дочернем процессе (на Windows используется spawn,
    поэтому состояние родителя не наследуется).
    """
    from metetl.logging_config import configure, get_logger
    configure()
    log = get_logger("worker")

    idx, painting_id, png_bytes, impl = payload
    pid = os.getpid()
    log.info("Convolution for image %d started (PID %d, impl=%s)", idx, pid, impl)

    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"image {idx}: failed to decode PNG in worker")

    kernel = _gaussian_kernel(5, 1.0)
    convolved = _DISPATCH[impl](img, kernel)
    if convolved.dtype != np.uint8:
        convolved = np.clip(convolved, 0, 255).astype(np.uint8)

    ok, encoded = cv2.imencode('.png', convolved)
    if not ok:
        raise RuntimeError(f"image {idx}: failed to encode PNG in worker")
    log.info("Convolution for image %d finished (PID %d)", idx, pid)
    return idx, painting_id, bytes(encoded)


class ImageProcessor:
    """Скачивание (aiohttp + aiofiles) и параллельная свёртка (ProcessPoolExecutor)."""

    def __init__(self, output_root: Path | str = "images", impl: ImplName = "numpy") -> None:
        if impl not in _VALID_IMPLS:
            raise ValueError(f"impl must be one of {_VALID_IMPLS}, got {impl!r}")
        self._output_root = Path(output_root)
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._impl: ImplName = impl
        _log.debug("ImageProcessor готов: output=%s, impl=%s", self._output_root, impl)

    @property
    def impl(self) -> ImplName:
        return self._impl

    @staticmethod
    def load_indexed(input_json: Path | str, count: int) -> List[Tuple[int, str]]:
        """
        Читает JSON со списком картин и возвращает [(idx, object_id), ...] длины count.
        Порядковый номер (idx) фиксируется здесь и далее не меняется.
        """
        path = Path(input_json)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        ids: List[str] = [str(item["object_id"]) for item in data]
        if not ids:
            _log.error("В %s нет картин", path)
            return []
        if count > len(ids):
            _log.warning("Запрошено %d, в JSON всего %d - усечём", count, len(ids))
            count = len(ids)
        random.shuffle(ids)
        indexed = list(enumerate(ids[:count], start=1))
        _log.info("Сформирован список URL: %s",
                  ", ".join(f"{i}->{pid}" for i, pid in indexed))
        return indexed

    def make_run_subdir(self, suffix: str = "") -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"run_{timestamp}{suffix}"
        subdir = self._output_root / name
        subdir.mkdir(parents=True, exist_ok=True)
        _log.info("Поддиректория для результатов: %s", subdir)
        return subdir

    # ------------------------------------------------------------------
    # async-этапы
    # ------------------------------------------------------------------

    async def _fetch_image_url(self, session: aiohttp.ClientSession,
                                idx: int, painting_id: str) -> Optional[str]:
        api_url = _MET_API.format(object_id=painting_id)
        try:
            async with session.get(api_url) as resp:
                if resp.status != 200:
                    _log.warning("Image %d: API HTTP %d for %s", idx, resp.status, painting_id)
                    return None
                data = await resp.json()
        except Exception as exc:
            _log.warning("Image %d: API error for %s: %s", idx, painting_id, exc)
            return None
        url = data.get("primaryImage")
        if not url:
            _log.warning("Image %d: no primaryImage for %s", idx, painting_id)
            return None
        return url

    async def _download_and_save_original(self, idx: int, painting_id: str,
                                            image_url: str,
                                            session: aiohttp.ClientSession,
                                            subdir: Path) -> Optional[bytes]:
        _log.info("Downloading image %d started", idx)
        try:
            async with session.get(image_url) as resp:
                if resp.status != 200:
                    _log.error("Image %d: HTTP %d", idx, resp.status)
                    return None
                raw_bytes = await resp.read()
        except Exception as exc:
            _log.error("Image %d: download failed: %s", idx, exc)
            return None

        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            _log.error("Image %d: decode failed", idx)
            return None
        ok, encoded = cv2.imencode('.png', img)
        if not ok:
            _log.error("Image %d: PNG encode failed", idx)
            return None
        png_bytes = bytes(encoded)

        original_path = subdir / f"{idx}_{painting_id}_original.png"
        async with aiofiles.open(original_path, "wb") as f:
            await f.write(png_bytes)
        _log.info("Downloading image %d finished -> %s", idx, original_path.name)
        return png_bytes

    @staticmethod
    async def _save_processed(idx: int, painting_id: str,
                                png_bytes: bytes, subdir: Path) -> None:
        path = subdir / f"{idx}_{painting_id}_processed.png"
        async with aiofiles.open(path, "wb") as f:
            await f.write(png_bytes)
        _log.info("Saving processed image %d finished -> %s", idx, path.name)

    async def _process_one(self, idx: int, painting_id: str,
                            session: aiohttp.ClientSession,
                            subdir: Path,
                            executor: ProcessPoolExecutor) -> bool:
        url = await self._fetch_image_url(session, idx, painting_id)
        if not url:
            return False
        png_bytes = await self._download_and_save_original(
            idx, painting_id, url, session, subdir)
        if png_bytes is None:
            return False

        loop = asyncio.get_running_loop()
        try:
            out_idx, out_id, out_bytes = await loop.run_in_executor(
                executor, _convolve_in_process,
                (idx, painting_id, png_bytes, self._impl))
        except Exception as exc:
            _log.error("Image %d: convolution failed: %s", idx, exc)
            return False

        await self._save_processed(out_idx, out_id, out_bytes, subdir)
        return True

    @timeit
    async def run_pipeline(self, indexed: Iterable[Tuple[int, str]],
                            subdir: Path) -> int:
        """gather-вариант: все элементы через _process_one параллельно."""
        with ProcessPoolExecutor() as executor:
            async with aiohttp.ClientSession() as session:
                tasks = [self._process_one(idx, pid, session, subdir, executor)
                         for idx, pid in indexed]
                results = await asyncio.gather(*tasks)
        success = sum(1 for r in results if r)
        _log.info("Конвейер завершён: успешно %d/%d", success, len(results))
        return success

    # ------------------------------------------------------------------
    # streaming-конвейер на async-генераторах (бонус Lab 4)
    # ------------------------------------------------------------------

    async def _download_stage(self, indexed: List[Tuple[int, str]],
                                session: aiohttp.ClientSession,
                                subdir: Path) -> AsyncIterator[DownloadPayload]:
        queue: asyncio.Queue = asyncio.Queue()

        async def one(idx: int, pid: str) -> None:
            url = await self._fetch_image_url(session, idx, pid)
            if not url:
                return
            png = await self._download_and_save_original(idx, pid, url, session, subdir)
            if png is not None:
                await queue.put((idx, pid, png))

        async def producer() -> None:
            await asyncio.gather(*(one(i, p) for i, p in indexed))
            await queue.put(None)

        prod_task = asyncio.create_task(producer())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            await prod_task

    async def _convolve_stage(self, source: AsyncIterator[DownloadPayload],
                                executor: ProcessPoolExecutor
                                ) -> AsyncIterator[DownloadPayload]:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        collectors: List[asyncio.Task] = []

        async def feeder() -> None:
            async for item in source:
                idx_local, _, _ = item
                fut = loop.run_in_executor(
                    executor, _convolve_in_process, (*item, self._impl))

                async def collect(f=fut, idx=idx_local) -> None:
                    try:
                        result = await f
                    except Exception as exc:
                        _log.error("Image %d: convolution failed: %s", idx, exc)
                        return
                    await queue.put(result)

                collectors.append(asyncio.create_task(collect()))
            if collectors:
                await asyncio.gather(*collectors, return_exceptions=True)
            await queue.put(None)

        feeder_task = asyncio.create_task(feeder())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            await feeder_task

    async def _save_stage(self, source: AsyncIterator[DownloadPayload],
                            subdir: Path) -> AsyncIterator[Tuple[int, str]]:
        async for idx, pid, png_bytes in source:
            await self._save_processed(idx, pid, png_bytes, subdir)
            yield idx, pid

    @timeit
    async def run_pipeline_streaming(self, indexed: List[Tuple[int, str]],
                                      subdir: Path) -> int:
        success = 0
        with ProcessPoolExecutor() as executor:
            async with aiohttp.ClientSession() as session:
                downloads = self._download_stage(indexed, session, subdir)
                convolutions = self._convolve_stage(downloads, executor)
                async for _idx, _pid in self._save_stage(convolutions, subdir):
                    success += 1
        _log.info("Streaming-конвейер завершён: успешно %d/%d", success, len(indexed))
        return success
