from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, Dict, Any
import json
import cv2


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
        return self._width, self._height

    @abstractmethod
    def apply_filter(self, kernel: np.ndarray) -> 'Artwork':
        """Абстрактный метод применения фильтра."""
        pass

    @abstractmethod
    def detect_edges_manual(self, threshold: Optional[int] = None) -> 'Artwork':
        """Абстрактный метод выделения границ."""
        pass

    @abstractmethod
    def detect_edges_biblio(self, threshold: Optional[int] = None) -> 'Artwork':
        """Абстрактный метод выделения границ (библиотечный)."""
        pass

    @abstractmethod
    def harris(self) -> 'Artwork':
        "Абстрактный метод Харисса"
        pass

    def __add__(self, other: 'Artwork') -> 'Artwork':
        """
        Перегрузка оператора сложения для объединения изображений.
        """
        if not isinstance(other, Artwork):
            raise TypeError("Можно складывать только объекты Artwork")

        if self._image.shape != other._image.shape:
            raise ValueError("Изображения должны иметь одинаковые размеры")

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
    Реализация для цветных изображений.
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

    def detect_edges_manual(self, threshold: Optional[int] = None) -> Artwork:
        """Выделение границ на цветном изображении."""
        gray = self.to_grayscale_manual()
        return gray.detect_edges_manual(threshold)

    def detect_edges_biblio(self, threshold: Optional[int] = None) -> Artwork:
        """Выделение границ на цветном изображении (библиотечная)."""
        gray = self.to_grayscale_biblio()
        return gray.detect_edges_manual(threshold)

    def harris(self):
        """Harris"""
        gray = self.to_grayscale_manual()
        return gray.harris()

    def to_grayscale_manual(self) -> 'GrayscaleArtwork':
        """Преобразование в оттенки серого."""
        coeffs = np.array([0.114, 0.587, 0.299])
        gray = np.dot(self._image, coeffs).astype(np.uint8)

        new_metadata = {**self._metadata, 'converted_to': 'grayscale'}
        return GrayscaleArtwork(gray, new_metadata)

    def to_grayscale_biblio(self) -> 'GrayscaleArtwork':
        """Преобразование в оттенки серого (библиотечная)."""
        gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)

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
    Реализация для черно-белых изображений.
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

    def detect_edges_manual(self, threshold: int = 100) -> 'GrayscaleArtwork':
        """Выделение границ методом Собеля с порогом."""
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        gx = self._convolution_manual(self._image, Kx)
        gy = self._convolution_manual(self._image, Ky)

        grad = np.sqrt(gx ** 2 + gy ** 2)
        grad_max = grad.max()
        if grad_max > 0:
            grad = grad / grad_max * 255

        edges = np.where(grad >= threshold, 255, 0).astype(np.uint8)

        new_metadata = {**self._metadata, 'edge_threshold': threshold}
        result = GrayscaleArtwork(edges, new_metadata)
        result._is_edge_detected = True
        return result

    def detect_edges_biblio(self, threshold1: int = 100, threshold2: int = 200) -> 'GrayscaleArtwork':
        """Выделение границ (библиотечная)"""
        canny_cv = cv2.Canny(self._image, threshold1, threshold2)

        new_metadata = {**self._metadata, 'threshold1': threshold1, 'threshold2': threshold2}
        result = GrayscaleArtwork(canny_cv, new_metadata)
        result._is_edge_detected = True
        return result

    def harris(self, block_size: int = 2, ksize: int = 5, k: float= 0.04) -> 'GrayscaleArtwork':
        harris_cv = cv2.cornerHarris(np.float32(self._image), block_size, ksize, k)
        harris_norm = cv2.normalize(harris_cv, None, 0, 255, cv2.NORM_MINMAX)
        harris_uint8 = harris_norm.astype(np.uint8)

        new_metadata = {**self._metadata, 'block_size': block_size, 'ksize': ksize, 'k': k}
        result = GrayscaleArtwork(harris_uint8, new_metadata)
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

    def to_color_artwork(self) -> 'ColorArtwork':
        """Преобразование в 3-x канальное."""
        rgb_image = np.repeat(self._image[:, :, np.newaxis], 3, axis=2)

        new_metadata = {**self._metadata, 'converted_from': 'grayscale'}
        return ColorArtwork(rgb_image, new_metadata)
