"""Скачивание и обработка изображений."""

from metetl.images.models import Artwork, ColorArtwork, GrayscaleArtwork
from metetl.images.processing import ImageProcessor

__all__ = ["Artwork", "ColorArtwork", "GrayscaleArtwork", "ImageProcessor"]
