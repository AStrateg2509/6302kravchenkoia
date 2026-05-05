from typing import Optional, Tuple, Dict, Any, List
import asyncio
import csv
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import aiofiles
import aiohttp
import cv2
import numpy as np

from artwork import Artwork, GrayscaleArtwork, ColorArtwork
from decor import timeit


def _convolve_in_process(payload: Tuple[int, str, bytes]) -> Tuple[int, str, bytes]:
    """
    Worker для ProcessPoolExecutor: декодирует PNG, применяет ручную свёртку
    с гауссовым ядром, возвращает PNG-байты результата. Должна быть на уровне
    модуля, иначе её нельзя пиклить для дочернего процесса.
    """
    idx, painting_id, png_bytes = payload
    pid = os.getpid()
    print(f"[LOG] Convolution for image {idx} started (PID {pid})", flush=True)

    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"image {idx}: failed to decode PNG in worker")

    size, sigma = 5, 1.0
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()

    if img.ndim == 3 and img.shape[2] == 3:
        artwork: Artwork = ColorArtwork(img, {"painting_id": painting_id})
    else:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        artwork = GrayscaleArtwork(img, {"painting_id": painting_id})

    processed = artwork.apply_filter(kernel)

    ok, encoded = cv2.imencode('.png', processed.image)
    if not ok:
        raise RuntimeError(f"image {idx}: failed to encode PNG in worker")
    print(f"[LOG] Convolution for image {idx} finished (PID {pid})", flush=True)
    return idx, painting_id, bytes(encoded)


class ImageProcessor:
    """
    Класс для обработки изображений.
    Управляет загрузкой, обработкой и сохранением изображений.
    """

    def __init__(self, input_dir: str = 'paintings', output_dir: str = 'filters'):
        """
        Инициализация процессора изображений.

        Args:
            input_dir: Директория для загруженных изображений
            output_dir: Директория для обработанных изображений
        """
        self._input_dir = Path(input_dir)
        self._output_dir = Path(output_dir)
        self._current_artwork: Optional[Artwork] = None

        # Создаем директории при необходимости
        self._input_dir.mkdir(exist_ok=True)
        self._output_dir.mkdir(exist_ok=True)

        print(f"[LOG] ImageProcessor инициализирован")
        print(f"[LOG] Входная директория: {self._input_dir}")
        print(f"[LOG] Выходная директория: {self._output_dir}")

    @property
    def current_artwork(self) -> Optional[Artwork]:
        """Свойство для получения текущего изображения."""
        return self._current_artwork

    @staticmethod
    def _gaussian_kernel(size: int = 5, sigma: float = 1) -> np.ndarray:
        """Создание гауссова ядра."""
        ax = np.linspace(-(size // 2), size // 2, size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        return kernel / np.sum(kernel)

    @timeit
    def download_image(self, csv_path: str, max_attempts=10) -> Optional[Artwork]:
        """
        Скачивание случайного изображения из метрополитен-музея.

        Args:
            csv_path: Путь к CSV файлу с данными о произведениях
            max_attempts: Максимальное число попыток скачки
        """
        print(f"[LOG] Начало скачивания изображения...")

        try:
            paintings = self._get_paintings(csv_path)
            if not paintings:
                print("[ERROR] Не найдено картин в CSV файле")
                return None

            artwork = None
            attempts = 0

            while artwork is None and attempts < max_attempts:
                attempts += 1
                artwork = self._try_download_painting(paintings)

            if artwork:
                self._current_artwork = artwork
                print(f"[LOG] Изображение успешно загружено: {artwork.dimensions}")
            else:
                print(f"[ERROR] Не удалось загрузить изображение после {max_attempts} попыток")

            return artwork

        except Exception as e:
            print(f"[ERROR] Ошибка при скачивании: {e}")
            return None

    @staticmethod
    def _get_paintings(csv_path: str) -> list:
        """Получение списка картин из CSV файла."""
        import csv

        paintings = []
        path = Path(csv_path)

        if not path.exists():
            print(f"[ERROR] Файл {csv_path} не найден")
            return paintings

        with open(path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Classification") == "Paintings":
                    object_id = row.get('Object ID')
                    if object_id:
                        paintings.append({
                            'object_number': row.get('Object Number'),
                            'object_id': object_id
                        })

        print(f"[LOG] Найдено {len(paintings)} картин в CSV")
        return paintings

    def _try_download_painting(self, paintings: list) -> Optional[Artwork]:
        """Попытка скачать одно изображение."""
        import random
        import requests
        import json

        painting = random.choice(paintings)
        object_id = painting["object_id"]

        url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
        response = requests.get(url)

        if response.status_code != 200:
            return None

        data = response.json()
        image_url = data.get("primaryImage")

        if not image_url:
            return None

        # Скачиваем изображение
        image_response = requests.get(image_url)
        if image_response.status_code != 200:
            return None

        # Сохраняем изображение и метаданные
        image_path = self._input_dir / f"{object_id}.jpg"
        json_path = self._input_dir / f"{object_id}.json"

        with open(image_path, "wb") as img_file:
            img_file.write(image_response.content)

        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

        # Загружаем изображение в память
        import cv2
        img = cv2.imread(str(image_path))

        # Определяем тип изображения и создаем соответствующий объект
        if len(img.shape) == 3 and img.shape[2] == 3:
            return ColorArtwork(img, data)
        else:
            return GrayscaleArtwork(img, data)

    @timeit
    def manual_grayscale(self, artwork: ColorArtwork) -> GrayscaleArtwork:
        """Ручное преобразование в оттенки серого."""
        print(f"[LOG] Применение manual grayscale...")
        return artwork.to_grayscale_manual()

    @timeit
    def opencv_grayscale(self, artwork: ColorArtwork) -> GrayscaleArtwork:
        """Преобразование в оттенки серого с помощью OpenCV."""
        print(f"[LOG] Применение OpenCV grayscale...")
        return artwork.to_grayscale_biblio()

    @timeit
    def manual_gaussian(self, artwork: Artwork, kernel_size: int = 5, sigma: float = 1) -> Artwork:
        """Ручное применение гауссова фильтра."""
        print(f"[LOG] Применение manual Gaussian...")
        kernel = self._gaussian_kernel(kernel_size, sigma)
        return artwork.apply_filter(kernel)

    @timeit
    def opencv_gaussian(self, artwork: Artwork, kernel_size: int = 5, sigma: float = 1) -> Artwork:
        """Применение гауссова фильтра с помощью OpenCV."""
        print(f"[LOG] Применение OpenCV Gaussian...")

        # Для ручной реализации используется apply_filter, для OpenCV - прямое применение
        if isinstance(artwork, ColorArtwork):
            # Для цветного изображения применяем GaussianBlur к каждому каналу
            blurred = cv2.GaussianBlur(artwork.image, (kernel_size, kernel_size), sigma)
            new_metadata = {**artwork.metadata, 'applied_filter': 'opencv_gaussian'}
            return ColorArtwork(blurred, new_metadata)
        else:
            # Для ч/б изображения
            blurred = cv2.GaussianBlur(artwork.image, (kernel_size, kernel_size), sigma)
            new_metadata = {**artwork.metadata, 'applied_filter': 'opencv_gaussian'}
            return GrayscaleArtwork(blurred, new_metadata)

    @timeit
    def manual_sobel(self, artwork: GrayscaleArtwork, threshold: int = 50) -> GrayscaleArtwork:
        """Ручное выделение границ методом Собеля."""
        print(f"[LOG] Применение manual Sobel...")
        return artwork.detect_edges_manual(threshold)

    @timeit
    def opencv_canny(self, artwork: GrayscaleArtwork, threshold1: int = 50, threshold2: int = 150) -> GrayscaleArtwork:
        """Выделение границ методом Canny с помощью OpenCV."""
        print(f"[LOG] Применение OpenCV Canny...")
        return artwork.detect_edges_biblio(threshold1, threshold2)

    @timeit
    def opencv_harris(self, artwork: GrayscaleArtwork, block_size: int = 2, ksize: int = 5,
                      k: float = 0.04) -> GrayscaleArtwork:
        """Выделение углов методом Harris с помощью OpenCV."""
        print(f"[LOG] Применение OpenCV Harris...")
        return artwork.harris(block_size, ksize, k)

    def process_image(self, artwork: Optional[Artwork] = None) -> Dict[str, Any]:
        """
        Обработка изображения различными фильтрами и сравнение методов.

        Args:
            artwork: Изображение для обработки (если None, используется текущее)

        Returns:
            Словарь с результатами обработки и временем выполнения
        """
        if artwork is None:
            artwork = self._current_artwork

        if artwork is None:
            print("[ERROR] Нет изображения для обработки")
            return {}

        print(f"[LOG] Начало сравнения методов обработки изображения...")
        print(f"[LOG] Тип изображения: {artwork.__class__.__name__}")

        results = {}

        gray_artwork = None
        if isinstance(artwork, ColorArtwork):
            gray_artwork = self.manual_grayscale(artwork)
        else:
            gray_artwork = artwork

        if isinstance(artwork, ColorArtwork):
            print("\n[LOG] === СРАВНЕНИЕ: Manual Grayscale vs OpenCV Grayscale ===")

            manual_gray = self.manual_grayscale(artwork)
            results['manual_grayscale'] = manual_gray

            opencv_gray = self.opencv_grayscale(artwork)
            results['opencv_grayscale'] = opencv_gray

            diff = np.abs(manual_gray.image.astype(np.int16) - opencv_gray.image.astype(np.int16))
            print(f"[LOG] Максимальная разница между методами grayscale: {diff.max()}")
            print(f"[LOG] Средняя разница: {diff.mean():.2f}")


        print("\n[LOG] === СРАВНЕНИЕ: Manual Gaussian vs OpenCV Gaussian ===")

        kernel_size = 5
        sigma = 1.0

        manual_gaussian = self.manual_gaussian(artwork, kernel_size, sigma)
        results['manual_gaussian'] = manual_gaussian

        opencv_gaussian = self.opencv_gaussian(artwork, kernel_size, sigma)
        results['opencv_gaussian'] = opencv_gaussian

        print("\n[LOG] === СРАВНЕНИЕ: Manual Sobel vs OpenCV Canny ===")

        threshold = 50

        manual_sobel = self.manual_sobel(gray_artwork, threshold)
        results['manual_sobel'] = manual_sobel

        opencv_canny = self.opencv_canny(gray_artwork, threshold, threshold * 3)
        results['opencv_canny'] = opencv_canny

        print("\n[LOG] === OpenCV Harris ===")

        opencv_harris = self.opencv_harris(gray_artwork)
        results['opencv_harris'] = opencv_harris

        print("\n[LOG] Сравнение методов завершено")
        return results

    def save_results(self, results: Dict[str, Artwork], base_name: str = "result"):
        """
        Сохранение результатов обработки.

        Args:
            results: Словарь с обработанными изображениями
            base_name: Базовое имя для сохраняемых файлов
        """
        import cv2

        print(f"[LOG] Сохранение результатов...")

        for name, artwork in results.items():
            filename = self._output_dir / f"{base_name}_{name}.jpg"
            cv2.imwrite(str(filename), artwork.image)
            print(f"[LOG] Сохранено: {filename}")

    def demonstrate_polymorphism(self):
        """
        Демонстрация полиморфизма - обработка разных типов изображений.
        """
        print("[LOG] ДЕМОНСТРАЦИЯ ПОЛИМОРФИЗМА")

        test_color = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        color_artwork = ColorArtwork(test_color, {"type": "test_color"})
        gray_artwork = GrayscaleArtwork(test_gray, {"type": "test_gray"})

        kernel = self._gaussian_kernel(3, 1)

        for artwork in [color_artwork, gray_artwork]:
            print(f"\nОбработка: {artwork.__class__.__name__}")
            print(f"  До обработки: {artwork.dimensions}")

            filtered = artwork.apply_filter(kernel)
            print(f"  После фильтра: {filtered.dimensions}")

            # Выделяем границы
            edges = artwork.detect_edges_manual(100)
            print(f"  Границы выделены: {'да' if hasattr(edges, 'has_edges') and edges.has_edges else 'нет'}")

            # Демонстрируем сложение
            if isinstance(artwork, ColorArtwork):
                # Для цветных складываем с копией
                blended = artwork + artwork
                print(f"  Сложение изображений: {blended.dimensions}")

    # ------------------------------------------------------------------
    # Lab 4: асинхронная загрузка + параллельная свёртка
    # ------------------------------------------------------------------

    @staticmethod
    def _load_painting_ids(csv_path: str) -> List[str]:
        """Возвращает список Object ID всех записей с Classification='Paintings'."""
        ids: List[str] = []
        path = Path(csv_path)
        if not path.exists():
            print(f"[ERROR] Файл {csv_path} не найден", flush=True)
            return ids
        with open(path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Classification") == "Paintings":
                    object_id = row.get('Object ID')
                    if object_id:
                        ids.append(object_id)
        print(f"[LOG] В CSV найдено {len(ids)} картин", flush=True)
        return ids

    async def _fetch_image_url(self, session: aiohttp.ClientSession,
                                idx: int, painting_id: str) -> Optional[str]:
        """Достать primaryImage URL из MET API."""
        api_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{painting_id}"
        try:
            async with session.get(api_url) as resp:
                if resp.status != 200:
                    print(f"[WARN] Image {idx}: API HTTP {resp.status} for {painting_id}", flush=True)
                    return None
                data = await resp.json()
        except Exception as exc:
            print(f"[WARN] Image {idx}: API error for {painting_id}: {exc}", flush=True)
            return None
        url = data.get("primaryImage")
        if not url:
            print(f"[WARN] Image {idx}: no primaryImage for {painting_id}", flush=True)
            return None
        return url

    async def _download_and_save_original(self, idx: int, painting_id: str,
                                            image_url: str,
                                            session: aiohttp.ClientSession,
                                            subdir: Path) -> Optional[bytes]:
        """Скачать байты картинки и сохранить как PNG через aiofiles."""
        print(f"[LOG] Downloading image {idx} started", flush=True)
        try:
            async with session.get(image_url) as resp:
                if resp.status != 200:
                    print(f"[ERROR] Image {idx}: HTTP {resp.status}", flush=True)
                    return None
                raw_bytes = await resp.read()
        except Exception as exc:
            print(f"[ERROR] Image {idx}: download failed: {exc}", flush=True)
            return None

        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[ERROR] Image {idx}: decode failed", flush=True)
            return None
        ok, encoded = cv2.imencode('.png', img)
        if not ok:
            print(f"[ERROR] Image {idx}: PNG encode failed", flush=True)
            return None
        png_bytes = bytes(encoded)

        original_path = subdir / f"{idx}_{painting_id}_original.png"
        async with aiofiles.open(original_path, 'wb') as f:
            await f.write(png_bytes)
        print(f"[LOG] Downloading image {idx} finished -> {original_path.name}", flush=True)
        return png_bytes

    @staticmethod
    async def _save_processed(idx: int, painting_id: str,
                                png_bytes: bytes, subdir: Path) -> None:
        """Сохранить результат свёртки через aiofiles."""
        path = subdir / f"{idx}_{painting_id}_processed.png"
        async with aiofiles.open(path, 'wb') as f:
            await f.write(png_bytes)
        print(f"[LOG] Saving processed image {idx} finished -> {path.name}", flush=True)

    async def _process_one(self, idx: int, painting_id: str,
                            session: aiohttp.ClientSession,
                            subdir: Path,
                            executor: ProcessPoolExecutor) -> bool:
        """Полный конвейер для одного изображения: API -> download -> conv -> save."""
        image_url = await self._fetch_image_url(session, idx, painting_id)
        if not image_url:
            return False

        png_bytes = await self._download_and_save_original(
            idx, painting_id, image_url, session, subdir)
        if png_bytes is None:
            return False

        loop = asyncio.get_running_loop()
        try:
            out_idx, out_id, out_bytes = await loop.run_in_executor(
                executor, _convolve_in_process, (idx, painting_id, png_bytes))
        except Exception as exc:
            print(f"[ERROR] Image {idx}: convolution failed: {exc}", flush=True)
            return False

        await self._save_processed(out_idx, out_id, out_bytes, subdir)
        return True

    async def run_pipeline(self, csv_path: str, count: int) -> Tuple[int, Path]:
        """
        Главный конвейер Lab 4. Формирует список из count картин с фиксированными
        порядковыми номерами, далее асинхронно скачивает их и параллельно
        обрабатывает свёрткой в отдельных процессах.
        """
        ids = self._load_painting_ids(csv_path)
        if not ids:
            print("[ERROR] Список картин пуст", flush=True)
            return 0, self._input_dir
        if count > len(ids):
            print(f"[WARN] Запрошено {count}, в CSV всего {len(ids)} - усечём", flush=True)
            count = len(ids)

        random.shuffle(ids)
        selected = ids[:count]
        # Порядковые номера фиксируются здесь и больше не меняются
        indexed = list(enumerate(selected, start=1))
        print(f"[LOG] Сформирован список URL: " +
              ", ".join(f"{i}->{pid}" for i, pid in indexed), flush=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        subdir = self._input_dir / f"run_{timestamp}"
        subdir.mkdir(parents=True, exist_ok=True)
        print(f"[LOG] Поддиректория для результатов: {subdir}", flush=True)

        with ProcessPoolExecutor() as executor:
            async with aiohttp.ClientSession() as session:
                tasks = [self._process_one(idx, pid, session, subdir, executor)
                         for idx, pid in indexed]
                results = await asyncio.gather(*tasks)

        success = sum(1 for r in results if r)
        print(f"[LOG] Конвейер завершён: успешно {success}/{count}", flush=True)
        return success, subdir

    # ------------------------------------------------------------------
    # Lab 4 (бонус): пайплайн на асинхронных генераторах.
    # Скачивание -> свёртка -> сохранение. Каждый этап - отдельный
    # async-генератор; стадии не блокируют друг друга, элементы текут
    # через пайплайн по мере готовности.
    # ------------------------------------------------------------------

    async def _download_stage(self, indexed: List[Tuple[int, str]],
                                session: aiohttp.ClientSession,
                                subdir: Path):
        """Этап 1: скачивает все картинки параллельно, yield-ит по мере готовности."""
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

    async def _convolve_stage(self, source, executor: ProcessPoolExecutor):
        """Этап 2: для каждого пришедшего элемента стартует свёртку в пуле
        процессов и yield-ит результат по мере готовности (не дожидаясь
        окончания всего входного потока)."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        collectors: List[asyncio.Task] = []

        async def feeder() -> None:
            async for item in source:
                fut = loop.run_in_executor(executor, _convolve_in_process, item)

                async def collect(f=fut) -> None:
                    try:
                        result = await f
                    except Exception as exc:
                        idx = item[0]
                        print(f"[ERROR] Image {idx}: convolution failed: {exc}", flush=True)
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

    async def _save_stage(self, source, subdir: Path):
        """Этап 3: сохраняет каждый пришедший результат через aiofiles
        и yield-ит индекс/id по мере записи."""
        async for idx, pid, png_bytes in source:
            await self._save_processed(idx, pid, png_bytes, subdir)
            yield idx, pid

    async def run_pipeline_streaming(self, csv_path: str, count: int) -> Tuple[int, Path]:
        """Главный конвейер на async-генераторах (бонус)."""
        ids = self._load_painting_ids(csv_path)
        if not ids:
            print("[ERROR] Список картин пуст", flush=True)
            return 0, self._input_dir
        if count > len(ids):
            print(f"[WARN] Запрошено {count}, в CSV всего {len(ids)} - усечём", flush=True)
            count = len(ids)

        random.shuffle(ids)
        selected = ids[:count]
        indexed = list(enumerate(selected, start=1))
        print(f"[LOG] Сформирован список URL: " +
              ", ".join(f"{i}->{pid}" for i, pid in indexed), flush=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        subdir = self._input_dir / f"run_{timestamp}_streaming"
        subdir.mkdir(parents=True, exist_ok=True)
        print(f"[LOG] Поддиректория для результатов: {subdir}", flush=True)

        success = 0
        with ProcessPoolExecutor() as executor:
            async with aiohttp.ClientSession() as session:
                downloads = self._download_stage(indexed, session, subdir)
                convolutions = self._convolve_stage(downloads, executor)
                async for _idx, _pid in self._save_stage(convolutions, subdir):
                    success += 1

        print(f"[LOG] Streaming-конвейер завершён: успешно {success}/{count}", flush=True)
        return success, subdir