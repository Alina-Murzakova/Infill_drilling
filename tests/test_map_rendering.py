from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from app.maps_handler.maps import Map


class MapRenderingTest(unittest.TestCase):
    def test_save_img_is_safe_from_background_threads(self) -> None:
        raster = Map(
            np.arange(100, dtype=float).reshape(10, 10),
            [100.0, 1.0, 0.0, 200.0, 0.0, -1.0],
            "",
            "test",
        )
        with tempfile.TemporaryDirectory() as temp:
            destinations = [Path(temp) / f"map-{number}.png" for number in range(2)]
            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(lambda path: raster.save_img(str(path)), destinations))
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in destinations))


if __name__ == "__main__":
    unittest.main()
