from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
from pathlib import Path

from artwork import Artwork, GrayscaleArtwork, ColorArtwork
from decor import timeit


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