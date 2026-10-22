"""YOLO inference wrapper. The only module that imports Ultralytics (lazily).

Everything downstream consumes plain ``Detection`` objects, so gesture logic and desktop
control can be unit-tested without a model, a GPU, or Ultralytics installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from src.config import ModelConfig


class ModelLoadError(RuntimeError):
    """Raised when weights are missing, unreadable, or trained on different classes."""


@dataclass(frozen=True)
class Detection:
    """One detected hand. Box coordinates are normalized to [0, 1] in the (mirrored) frame."""

    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def anchor(self, anchor_y: float) -> tuple[float, float]:
        """Approximate fingertip position: horizontally centred, ``anchor_y`` of the height below the top.

        This is a heuristic. A bounding box does not contain finger-joint positions.
        """
        return ((self.x1 + self.x2) / 2.0, self.y1 + anchor_y * self.height)


_MISSING_WEIGHTS_HELP = """Model weights not found: {path}

AirCommand needs weights fine-tuned on its gesture classes; generic COCO weights
(yolo11n.pt) have no hand classes and will not work.

1. Train with training/aircommand_training.ipynb in Google Colab (see README, section "Training").
2. Download best.pt from the notebook's last cell.
3. Save it as models/best.pt (or pass --model path/to/best.pt).

The live app cannot run without trained weights. Unit tests (python -m pytest) and the
fake-detector pipeline test do not need weights.
"""


def _resolve_device(device: str) -> Optional[str]:
    return None if device == "auto" else device


class GestureDetector:
    """Loads a fine-tuned Ultralytics YOLO model and returns normalized ``Detection`` lists."""

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        path = Path(cfg.path)
        if not path.is_file():
            raise ModelLoadError(_MISSING_WEIGHTS_HELP.format(path=path))
        try:
            from ultralytics import YOLO  # imported lazily on purpose
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ModelLoadError(
                "Ultralytics is not installed in this environment. Activate the virtualenv and run:\n"
                "  pip install -r requirements.txt"
            ) from exc
        try:
            self.model: Any = YOLO(str(path))
        except Exception as exc:  # corrupted file, incompatible ultralytics version, wrong format
            raise ModelLoadError(
                f"Failed to load {path}: {exc}\n"
                "- Re-download best.pt (partial downloads are a common cause).\n"
                "- Use the same Ultralytics major/minor version locally as in Colab (see requirements.txt)."
            ) from exc

        names = self.model.names  # {id: name}
        model_classes = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        if list(model_classes) != list(cfg.classes):
            raise ModelLoadError(
                "Class list in the weights does not match config/gestures.yaml model.classes.\n"
                f"  weights: {model_classes}\n  config:  {list(cfg.classes)}\n"
                "Either fix model.classes (order matters) or retrain with matching classes."
            )
        self.class_names: tuple[str, ...] = tuple(model_classes)
        self._device = _resolve_device(cfg.device)

    def warmup(self, shape: tuple[int, int] = (480, 640)) -> None:
        dummy = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        self.detect(dummy)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame_bgr,
            conf=self.cfg.min_threshold,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self._device,
            verbose=False,
        )
        if not results:
            return []
        return self.parse_boxes(results[0].boxes, self.class_names)

    @staticmethod
    def parse_boxes(boxes: Any, class_names: Sequence[str]) -> list[Detection]:
        """Convert an Ultralytics ``Boxes`` object to ``Detection`` objects."""
        if boxes is None or len(boxes) == 0:
            return []
        xyxyn = boxes.xyxyn.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        out: list[Detection] = []
        for (x1, y1, x2, y2), conf, cls in zip(xyxyn, confs, clss):
            if 0 <= cls < len(class_names):
                out.append(Detection(class_names[cls], float(conf), float(x1), float(y1), float(x2), float(y2)))
        return out
