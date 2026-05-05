"""Тесты класса Artwork и его подклассов."""

import unittest

import numpy as np

from metetl.images.models import ColorArtwork, GrayscaleArtwork


class TestColorArtwork(unittest.TestCase):
    def setUp(self) -> None:
        self.rgb = np.full((4, 4, 3), 100, dtype=np.uint8)
        self.metadata = {"object_id": "1"}
        self.artwork = ColorArtwork(self.rgb, self.metadata)

    def test_dimensions(self) -> None:
        self.assertEqual(self.artwork.dimensions, (4, 4))

    def test_to_grayscale_manual(self) -> None:
        gray = self.artwork.to_grayscale_manual()
        self.assertIsInstance(gray, GrayscaleArtwork)
        self.assertEqual(gray.image.shape, (4, 4))
        self.assertEqual(gray.image.dtype, np.uint8)

    def test_apply_filter_identity(self) -> None:
        kernel = np.zeros((3, 3), dtype=np.float32)
        kernel[1, 1] = 1.0
        out = self.artwork.apply_filter(kernel)
        self.assertEqual(out.image.shape, self.rgb.shape)
        np.testing.assert_array_equal(out.image, self.rgb)

    def test_add_blends_two_images(self) -> None:
        other = ColorArtwork(np.full((4, 4, 3), 200, dtype=np.uint8), {})
        blend = self.artwork + other
        self.assertEqual(blend.image.dtype, np.uint8)
        self.assertTrue(np.all(blend.image == 150))

    def test_invalid_shape_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ColorArtwork(np.zeros((4, 4), dtype=np.uint8), {})


class TestGrayscaleArtwork(unittest.TestCase):
    def test_apply_filter_changes_metadata(self) -> None:
        img = np.full((6, 6), 128, dtype=np.uint8)
        gray = GrayscaleArtwork(img, {"object_id": "1"})
        kernel = np.full((3, 3), 1 / 9, dtype=np.float32)
        out = gray.apply_filter(kernel)
        self.assertIn("applied_filter", out.metadata)

    def test_detect_edges_manual_marks_flag(self) -> None:
        img = np.zeros((10, 10), dtype=np.uint8)
        img[5:, :] = 255
        gray = GrayscaleArtwork(img, {})
        edges = gray.detect_edges_manual(threshold=50)
        self.assertTrue(edges.has_edges)
        self.assertEqual(edges.image.shape, img.shape)
        self.assertGreater(int(edges.image.sum()), 0)

    def test_invalid_shape_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GrayscaleArtwork(np.zeros((4, 4, 3), dtype=np.uint8), {})


if __name__ == "__main__":
    unittest.main()
