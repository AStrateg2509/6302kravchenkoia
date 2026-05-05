"""Тесты класса ImageProcessor + интеграционный тест с Mock-Artwork."""

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from metetl.images.processing import ImageProcessor, _convolve_numpy


class TestImageProcessor(unittest.TestCase):
    def test_invalid_impl_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                ImageProcessor(output_root=tmp, impl="bogus")

    def test_load_indexed_reads_json_and_truncates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "to_download.json"
            json_path.write_text(json.dumps([
                {"object_id": "111"},
                {"object_id": "222"},
                {"object_id": "333"},
            ]))
            indexed = ImageProcessor.load_indexed(json_path, count=2)
            self.assertEqual(len(indexed), 2)
            ids = [pid for _, pid in indexed]
            self.assertTrue(set(ids).issubset({"111", "222", "333"}))
            indices = [idx for idx, _ in indexed]
            self.assertEqual(indices, [1, 2])

    def test_make_run_subdir_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = ImageProcessor(output_root=tmp, impl="numpy")
            subdir = proc.make_run_subdir(suffix="_test")
            self.assertTrue(subdir.is_dir())
            self.assertTrue(subdir.name.startswith("run_"))
            self.assertTrue(subdir.name.endswith("_test"))

    def test_convolve_numpy_dispatches_to_artwork(self) -> None:
        """Mock-Artwork: подменяем ColorArtwork внутри processing,
        проверяем, что _convolve_numpy зовёт apply_filter и возвращает image."""
        fake_image = np.zeros((4, 4, 3), dtype=np.uint8)
        kernel = np.eye(3, dtype=np.float32)

        result_artwork = MagicMock()
        result_artwork.image = np.full((4, 4, 3), 42, dtype=np.uint8)

        mock_artwork = MagicMock()
        mock_artwork.apply_filter.return_value = result_artwork

        with patch("metetl.images.processing.ColorArtwork",
                   return_value=mock_artwork) as mock_cls:
            out = _convolve_numpy(fake_image, kernel)

        mock_cls.assert_called_once()
        mock_artwork.apply_filter.assert_called_once()
        np.testing.assert_array_equal(out, result_artwork.image)


class TestProcessOneIntegration(unittest.IsolatedAsyncioTestCase):
    """Интеграционный тест _process_one: aiohttp, aiofiles, ProcessPoolExecutor — все замоканы."""

    async def test_full_path_with_mocked_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subdir = Path(tmp) / "run_test"
            subdir.mkdir()
            proc = ImageProcessor(output_root=tmp, impl="numpy")

            # 1×1 PNG bytes — корректный заголовок, чтобы cv2.imdecode не упал
            import cv2
            ok, encoded = cv2.imencode(".png",
                                          np.zeros((4, 4, 3), dtype=np.uint8))
            assert ok
            png_bytes = bytes(encoded)

            api_resp = AsyncMock()
            api_resp.status = 200
            api_resp.json = AsyncMock(return_value={"primaryImage": "http://x/y.jpg"})
            api_resp.__aenter__ = AsyncMock(return_value=api_resp)
            api_resp.__aexit__ = AsyncMock(return_value=False)

            img_resp = AsyncMock()
            img_resp.status = 200
            img_resp.read = AsyncMock(return_value=png_bytes)
            img_resp.__aenter__ = AsyncMock(return_value=img_resp)
            img_resp.__aexit__ = AsyncMock(return_value=False)

            session = MagicMock()
            session.get = MagicMock(side_effect=[api_resp, img_resp])

            with patch("metetl.images.processing._convolve_in_process",
                       return_value=(1, "999", png_bytes)) as mock_conv:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    ok = await proc._process_one(1, "999", session, subdir, executor)

            self.assertTrue(ok)
            mock_conv.assert_called_once()
            self.assertTrue((subdir / "1_999_original.png").exists())
            self.assertTrue((subdir / "1_999_processed.png").exists())


if __name__ == "__main__":
    unittest.main()
