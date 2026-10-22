"""Webcam / video-file capture with actionable errors and an optional latest-frame reader thread."""

from __future__ import annotations

import platform
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import CameraConfig


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or stops delivering frames."""


_BACKENDS = {
    "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
    "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
    "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
    "any": cv2.CAP_ANY,
}


def _backend_order(name: str) -> list[tuple[str, int]]:
    if name != "auto":
        return [(name, _BACKENDS[name])]
    system = platform.system()
    if system == "Windows":
        names = ["dshow", "msmf", "any"]  # DirectShow usually opens fastest on Windows
    elif system == "Linux":
        names = ["v4l2", "any"]
    else:
        names = ["any"]
    return [(n, _BACKENDS[n]) for n in names]


def _camera_help(index: int) -> str:
    system = platform.system()
    tips = [
        f"Could not open camera index {index}.",
        "- Close other apps using the webcam (Zoom, Teams, browser tabs, OBS).",
        "- Try another index: python -m src.main --camera 1",
    ]
    if system == "Windows":
        tips += ["- Settings > Privacy & security > Camera: allow camera access for desktop apps.",
                 "- Try camera.backend: msmf (or dshow) in config/gestures.yaml."]
    elif system == "Linux":
        tips += ["- List devices: ls /dev/video*   (v4l2-ctl --list-devices gives names)",
                 "- Permission: sudo usermod -aG video $USER, then log out and back in.",
                 "- Try camera.backend: any in config/gestures.yaml."]
    tips.append("- No webcam? Test the pipeline with a video file: python -m src.main --source path/to/video.mp4 --dry-run")
    return "\n".join(tips)


class _LatestFrameReader:
    """Background thread that keeps only the newest frame so inference never processes stale frames."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._failures = 0
        self._running = True
        self._thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            with self._lock:
                if ok and frame is not None:
                    self._frame = frame
                    self._seq += 1
                    self._failures = 0
                else:
                    self._failures += 1
            if not ok:
                time.sleep(0.01)

    def latest(self, last_seq: int, timeout_s: float) -> tuple[Optional[np.ndarray], int, int]:
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                if self._seq != last_seq and self._frame is not None:
                    return self._frame, self._seq, self._failures
                failures = self._failures
            if time.monotonic() >= deadline:
                return None, last_seq, failures
            time.sleep(0.002)

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


class Camera:
    """Context-managed frame source. Use ``source`` for a video file instead of a webcam."""

    MAX_CONSECUTIVE_FAILURES = 30

    def __init__(self, cfg: CameraConfig, source: Optional[str] = None) -> None:
        self.cfg = cfg
        self.source = source
        self.is_file = source is not None
        self._cap: Optional[cv2.VideoCapture] = None
        self._reader: Optional[_LatestFrameReader] = None
        self._seq = 0
        self._failures = 0
        self.backend_name = "file" if self.is_file else ""

    def open(self) -> "Camera":
        if self.is_file:
            path = Path(self.source)  # type: ignore[arg-type]
            if not path.is_file():
                raise CameraError(f"Video file not found: {path}")
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise CameraError(f"OpenCV could not decode {path}. Try an .mp4 (H.264) or .avi file.")
            self._cap = cap
            return self

        last_err = ""
        for name, api in _backend_order(self.cfg.backend):
            cap = cv2.VideoCapture(self.cfg.index, api)
            if not cap.isOpened():
                cap.release()
                last_err = f"backend {name} failed to open"
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok = False
            for _ in range(20):  # some drivers return a few empty frames while warming up
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                time.sleep(0.05)
            if not ok:
                cap.release()
                last_err = f"backend {name} opened but returned no frames"
                continue
            self._cap = cap
            self.backend_name = name
            if self.cfg.threaded:
                self._reader = _LatestFrameReader(cap)
            return self
        raise CameraError(f"{_camera_help(self.cfg.index)}\n(last error: {last_err})")

    @property
    def file_fps(self) -> float:
        """Nominal FPS of a video file (used as the clock for replay so hold/swipe timings stay real)."""
        if self._cap is None:
            return float(self.cfg.fps)
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return fps if 1.0 <= fps <= 240.0 else float(self.cfg.fps)

    @property
    def resolution(self) -> tuple[int, int]:
        if self._cap is None:
            return (0, 0)
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self) -> Optional[np.ndarray]:
        """Return the next frame (mirrored if configured). None means end of a video file."""
        if self._cap is None:
            raise CameraError("Camera.read() called before open().")
        if self._reader is not None:
            frame, self._seq, failures = self._reader.latest(self._seq, timeout_s=2.0)
            if frame is None:
                raise CameraError(
                    f"Camera stopped delivering frames ({failures} failed reads). "
                    "It may have been unplugged or grabbed by another application."
                )
            frame = frame.copy()
        else:
            while True:
                ok, frame = self._cap.read()
                if ok and frame is not None:
                    self._failures = 0
                    break
                if self.is_file:
                    return None
                self._failures += 1
                if self._failures >= self.MAX_CONSECUTIVE_FAILURES:
                    raise CameraError("Camera stopped delivering frames. Check the cable / other apps.")
                time.sleep(0.01)
        if self.cfg.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.release()
