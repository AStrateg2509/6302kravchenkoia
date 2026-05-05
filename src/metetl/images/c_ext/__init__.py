from __future__ import annotations

import numpy as np

try:
    from . import image_ops
except Exception:
    image_ops = None


def is_available() -> bool:
    return image_ops is not None


def convolve_uint8(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if image_ops is None:
        raise RuntimeError("C extension is not available")

    image = np.asarray(image, dtype=np.uint8)
    kernel = np.asarray(kernel, dtype=np.float32)

    if image.ndim == 2:
        contiguous_image = np.ascontiguousarray(image)
        contiguous_kernel = np.ascontiguousarray(kernel)
        raw = image_ops.convolve_gray_u8(contiguous_image, contiguous_kernel)
        return np.frombuffer(raw, dtype=np.float32).reshape(contiguous_image.shape).copy()

    if image.ndim == 3:
        channels = [convolve_uint8(image[..., channel], kernel) for channel in range(image.shape[2])]
        return np.stack(channels, axis=2)

    raise ValueError("image must be 2D or 3D")


__all__ = ["image_ops", "is_available", "convolve_uint8"]
