"""Detector tests that need neither Ultralytics nor weights."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.detector import Detection, GestureDetector, ModelLoadError
from tests._helpers import default_config


class _Arr:
    """Mimics a torch tensor's .cpu().numpy() chain."""

    def __init__(self, a):
        self._a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Boxes:
    def __init__(self, xyxyn, conf, cls):
        self.xyxyn, self.conf, self.cls = _Arr(xyxyn), _Arr(conf), _Arr(cls)

    def __len__(self):
        return len(self.conf.numpy())


class TestDetector(unittest.TestCase):
    def test_missing_weights_gives_actionable_error(self):
        cfg = replace(default_config().model, path=Path(tempfile.gettempdir()) / "definitely_missing_best.pt")
        with self.assertRaises(ModelLoadError) as ctx:
            GestureDetector(cfg)
        self.assertIn("models/best.pt", str(ctx.exception))

    def test_parse_boxes(self):
        names = ("one", "fist")
        boxes = _Boxes([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.9, 0.9], [0, 0, 1, 1]], [0.9, 0.7, 0.5], [0, 1, 7])
        dets = GestureDetector.parse_boxes(boxes, names)
        self.assertEqual([d.label for d in dets], ["one", "fist"])  # unknown class id 7 dropped
        self.assertAlmostEqual(dets[0].x2, 0.3)

    def test_parse_empty(self):
        self.assertEqual(GestureDetector.parse_boxes(_Boxes(np.zeros((0, 4)), [], []), ("one",)), [])
        self.assertEqual(GestureDetector.parse_boxes(None, ("one",)), [])

    def test_detection_geometry(self):
        d = Detection("one", 0.9, 0.2, 0.2, 0.4, 0.6)
        self.assertEqual(d.center, (0.30000000000000004, 0.4))
        ax, ay = d.anchor(0.25)
        self.assertAlmostEqual(ax, 0.3)
        self.assertAlmostEqual(ay, 0.3)
        self.assertAlmostEqual(d.area, 0.08)


if __name__ == "__main__":
    unittest.main()
