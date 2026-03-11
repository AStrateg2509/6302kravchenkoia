from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
from pathlib import Path
import json


class Artwork(ABC):
    """
    Абстрактный базовый класс для художественных произведений.
    Инкапсулирует изображение и метаданные.
    """
    __slots__ = ['_image', '_metadata', '_width', '_height']

    def __init__(self, image: np.ndarray, metadata: Dict[str, Any]):
        """
        Инициализация художественного произведения.

        Args:
            image: Изображение в виде numpy массива
            metadata: Метаданные изображения
        """
        self._image = image
        self._metadata = metadata
        self._width = image.shape[1]
        self._height = image.shape[0]

    @property
    def image(self) -> np.ndarray:
        """Геттер для изображения."""
        return self._image.copy()

    @property
    def metadata(self) -> Dict[str, Any]:
        """Геттер для метаданных."""
        return self._metadata.copy()

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Свойство возвращает размеры изображения."""
        return (self._width, self._height)

    @abstractmethod
    def apply_filter(self, kernel: np.ndarray) -> 'Artwork':
        """Абстрактный метод применения фильтра."""
        pass

    @abstractmethod
    def detect_edges(self, threshold: Optional[int] = None) -> 'Artwork':
        """Абстрактный метод выделения границ."""
        pass

    def __add__(self, other: 'Artwork') -> 'Artwork':
        """
        Перегрузка оператора сложения для объединения изображений.
        """
        if not isinstance(other, Artwork):
            raise TypeError("Можно складывать только объекты Artwork")

        if self._image.shape != other._image.shape:
            raise ValueError("Изображения должны иметь одинаковые размеры")

        # Среднее арифметическое двух изображений
        blended = ((self._image.astype(np.float32) +
                    other._image.astype(np.float32)) / 2).astype(np.uint8)

        merged_metadata = {
            'source1': self._metadata,
            'source2': other._metadata,
            'blend_type': 'average'
        }

        return self.__class__(blended, merged_metadata)

    def __str__(self) -> str:
        """Перегрузка метода преобразования в строку."""
        return (f"{self.__class__.__name__}:\n"
                f"  Размер: {self._width}x{self._height}\n"
                f"  Формат: {self._image.dtype}\n"
                f"  Метаданные: {json.dumps(self._metadata, indent=2, ensure_ascii=False)}")

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        """Статический метод для преобразования в uint8."""
        return np.clip(img, 0, 255).astype(np.uint8)


class ColorArtwork(Artwork):
    """
    Конкретная реализация для цветных изображений.
    """

    def __init__(self, image: np.ndarray, metadata: Dict[str, Any]):
        """Инициализация цветного изображения."""
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Цветное изображение должно иметь 3 канала")
        super().__init__(image, metadata)

    def apply_filter(self, kernel: np.ndarray) -> 'ColorArtwork':
        """Применение фильтра к цветному изображению."""
        # Применяем фильтр к каждому каналу отдельно
        filtered_channels = []
        for i in range(3):
            channel = self._image[:, :, i]
            filtered = self._convolution_manual(channel, kernel)
            filtered_channels.append(filtered)

        filtered_image = np.stack(filtered_channels, axis=2)
        filtered_image = self._to_uint8(filtered_image)

        new_metadata = {**self._metadata, 'applied_filter': 'color_filter'}
        return ColorArtwork(filtered_image, new_metadata)

    def detect_edges(self, threshold: Optional[int] = None) -> Artwork:
        """Выделение границ на цветном изображении."""
        # Конвертируем в оттенки серого для выделения границ
        gray = self._to_grayscale()
        return gray.detect_edges(threshold)

    def _to_grayscale(self) -> 'GrayscaleArtwork':
        """Преобразование в оттенки серого."""
        coeffs = np.array([0.114, 0.587, 0.299])
        gray = np.dot(self._image, coeffs).astype(np.uint8)

        new_metadata = {**self._metadata, 'converted_to': 'grayscale'}
        return GrayscaleArtwork(gray, new_metadata)

    @staticmethod
    def _convolution_manual(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Ручная реализация свертки."""
        from numpy.lib.stride_tricks import as_strided

        h, w = img.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

        shape = (h, w, kh, kw)
        strides = padded.strides * 2
        windows = as_strided(padded, shape=shape, strides=strides)

        return np.tensordot(windows, kernel, axes=([2, 3], [0, 1]))


class GrayscaleArtwork(Artwork):
    """
    Конкретная реализация для черно-белых изображений.
    """
    __slots__ = ['_is_edge_detected']

    def __init__(self, image: np.ndarray, metadata: Dict[str, Any]):
        """Инициализация черно-белого изображения."""
        if len(image.shape) != 2:
            raise ValueError("Ч/б изображение должно иметь 2 измерения")
        self._is_edge_detected = False
        super().__init__(image, metadata)

    def apply_filter(self, kernel: np.ndarray) -> 'GrayscaleArtwork':
        """Применение фильтра к ч/б изображению."""
        filtered = self._convolution_manual(self._image, kernel)
        filtered = self._to_uint8(filtered)

        new_metadata = {**self._metadata, 'applied_filter': 'grayscale_filter'}
        return GrayscaleArtwork(filtered, new_metadata)

    def detect_edges(self, threshold: int = 100) -> 'GrayscaleArtwork':
        """Выделение границ методом Собеля с порогом."""
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        gx = self._convolution_manual(self._image, Kx)
        gy = self._convolution_manual(self._image, Ky)

        grad = np.sqrt(gx ** 2 + gy ** 2)
        grad_max = grad.max()
        if grad_max > 0:
            grad = grad / grad_max * 255

        # Применяем порог
        edges = np.where(grad >= threshold, 255, 0).astype(np.uint8)

        new_metadata = {**self._metadata, 'edge_threshold': threshold}
        result = GrayscaleArtwork(edges, new_metadata)
        result._is_edge_detected = True
        return result

    @property
    def has_edges(self) -> bool:
        """Свойство, показывающее, выделены ли границы."""
        return self._is_edge_detected

    @staticmethod
    def _convolution_manual(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Ручная реализация свертки."""
        from numpy.lib.stride_tricks import as_strided

        h, w = img.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

        shape = (h, w, kh, kw)
        strides = padded.strides * 2
        windows = as_strided(padded, shape=shape, strides=strides)

        return np.tensordot(windows, kernel, axes=([2, 3], [0, 1]))


def timeit(func):
    """
    Декоратор для замера времени выполнения функции.
    """
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIME] {func.__name__} выполнена за {end - start:.4f} секунд")
        return result

    return wrapper


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
    def download_image(self, csv_path: str) -> Optional[Artwork]:
        """
        Скачивание случайного изображения из метрополитен-музея.

        Args:
            csv_path: Путь к CSV файлу с данными о произведениях
        """
        print(f"[LOG] Начало скачивания изображения...")

        try:
            # Получаем список картин
            paintings = self._get_paintings(csv_path)
            if not paintings:
                print("[ERROR] Не найдено картин в CSV файле")
                return None

            # Пытаемся скачать изображение
            artwork = None
            attempts = 0
            max_attempts = 10

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

    def _get_paintings(self, csv_path: str) -> list:
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
    def process_image(self, artwork: Optional[Artwork] = None) -> Dict[str, Artwork]:
        """
        Обработка изображения различными фильтрами.

        Args:
            artwork: Изображение для обработки (если None, используется текущее)

        Returns:
            Словарь с обработанными изображениями
        """
        if artwork is None:
            artwork = self._current_artwork

        if artwork is None:
            print("[ERROR] Нет изображения для обработки")
            return {}

        print(f"[LOG] Начало обработки изображения...")

        results = {}

        # Гауссов фильтр
        kernel = self._gaussian_kernel(5, 1)
        results['gaussian'] = artwork.apply_filter(kernel)
        print(f"[LOG] Применен гауссов фильтр")

        # Выделение границ
        if isinstance(artwork, ColorArtwork):
            # Для цветного сначала конвертируем в ч/б
            gray = artwork._to_grayscale()
            results['edges'] = gray.detect_edges(50)
        else:
            results['edges'] = artwork.detect_edges(50)
        print(f"[LOG] Выделены границы")

        # Комбинация фильтров
        results['gaussian_edges'] = results['gaussian'].detect_edges(50)
        print(f"[LOG] Применена комбинация фильтров")

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
        print("\n" + "=" * 50)
        print("ДЕМОНСТРАЦИЯ ПОЛИМОРФИЗМА")
        print("=" * 50)

        # Создаем тестовые изображения
        test_color = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        color_artwork = ColorArtwork(test_color, {"type": "test_color"})
        gray_artwork = GrayscaleArtwork(test_gray, {"type": "test_gray"})

        kernel = self._gaussian_kernel(3, 1)

        # Демонстрируем полиморфное поведение
        for artwork in [color_artwork, gray_artwork]:
            print(f"\nОбработка: {artwork.__class__.__name__}")
            print(f"  До обработки: {artwork.dimensions}")

            # Применяем фильтр (метод работает по-разному для разных классов)
            filtered = artwork.apply_filter(kernel)
            print(f"  После фильтра: {filtered.dimensions}")

            # Выделяем границы
            edges = artwork.detect_edges(100)
            print(f"  Границы выделены: {'да' if hasattr(edges, 'has_edges') and edges.has_edges else 'нет'}")

            # Демонстрируем сложение
            if isinstance(artwork, ColorArtwork):
                # Для цветных складываем с копией
                blended = artwork + artwork
                print(f"  Сложение изображений: {blended.dimensions}")


def main():
    """
    Основная функция для демонстрации работы классов.
    """
    # Создаем процессор изображений
    processor = ImageProcessor()

    # Скачиваем изображение
    artwork = processor.download_image('MetObjects.csv')

    if artwork:
        print("\n" + str(artwork) + "\n")

        # Обрабатываем изображение
        results = processor.process_image(artwork)

        # Сохраняем результаты
        if results:
            object_id = artwork.metadata.get('objectID', 'unknown')
            processor.save_results(results, f"painting_{object_id}")

        # Демонстрируем полиморфизм
        processor.demonstrate_polymorphism()
    else:
        print("Не удалось загрузить изображение")


if __name__ == "__main__":
    main()