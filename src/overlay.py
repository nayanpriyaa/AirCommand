"""On-screen visualisation: state banner, detections, hold progress, swipe trail, help."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from src.detector import Detection
from src.gesture_logic import EngineStatus, StableGesture
from src.safety import SafetyState

FONT = cv2.FONT_HERSHEY_SIMPLEX

STATE_COLORS = {  # BGR
    SafetyState.INACTIVE: (90, 90, 90),
    SafetyState.ACTIVE: (60, 170, 40),
    SafetyState.ESTOP: (40, 40, 210),
}

_PALETTE = [(255, 170, 0), (0, 140, 255), (200, 80, 255), (0, 210, 210), (120, 220, 90), (160, 160, 160)]

HELP_LINES = [
    "Esc  EMERGENCY STOP (global)",
    "F8   arm / disarm (global)",
    "F9   clear e-stop (global)",
    "a/d  arm / disarm   r  clear e-stop",
    "s    save screenshot",
    "h    toggle help    q  quit",
]


@dataclass
class OverlayData:
    state: SafetyState
    message: str
    message_age: float
    detections: Sequence[Detection]
    primary: Optional[Detection]
    stable: StableGesture
    status: EngineStatus
    roles: dict[str, Optional[str]]
    class_names: Sequence[str]
    active_region: tuple[float, float, float, float]
    anchor_y: float
    fps: float
    infer_ms: float
    last_event: str
    last_event_age: float
    dry_run: bool
    show_help: bool
    hotkeys_ok: bool


def _panel(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.55) -> None:
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)


def _text(img: np.ndarray, s: str, org: tuple[int, int], scale: float = 0.5,
          color: tuple[int, int, int] = (255, 255, 255), thick: int = 1) -> None:
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _bar(img: np.ndarray, x: int, y: int, w: int, label: str, frac: float, color: tuple[int, int, int]) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + 12), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * frac), y + 12), color, -1)
    _text(img, label, (x + w + 8, y + 11), 0.42)


def draw(frame: np.ndarray, d: OverlayData) -> np.ndarray:
    img = frame
    h, w = img.shape[:2]
    color_of = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(d.class_names)}

    # Active region (only meaningful while pointing)
    if d.stable.label == d.roles.get("pointer"):
        x1, y1, x2, y2 = d.active_region
        cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (200, 200, 0), 1, cv2.LINE_AA)

    # Detections
    for det in d.detections:
        is_primary = det is d.primary
        col = color_of.get(det.label, (200, 200, 200)) if is_primary else (150, 150, 150)
        p1, p2 = (int(det.x1 * w), int(det.y1 * h)), (int(det.x2 * w), int(det.y2 * h))
        cv2.rectangle(img, p1, p2, col, 3 if is_primary else 1, cv2.LINE_AA)
        _text(img, f"{det.label} {det.confidence:.2f}", (p1[0], max(52, p1[1] - 6)), 0.5, col)
    if d.primary is not None and d.primary.label == d.roles.get("pointer"):
        ax, ay = d.primary.anchor(d.anchor_y)
        cv2.circle(img, (int(ax * w), int(ay * h)), 6, (0, 255, 255), -1, cv2.LINE_AA)

    # Swipe trail
    trail = d.status.swipe_trail
    if len(trail) >= 2:
        pts = np.array([[int(x * w), int(y * h)] for x, y in trail], dtype=np.int32)
        cv2.polylines(img, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)

    # Banner
    cv2.rectangle(img, (0, 0), (w, 34), STATE_COLORS[d.state], -1)
    title = f"AirCommand  {d.state.value}" + ("  [DRY-RUN]" if d.dry_run else "")
    _text(img, title, (10, 24), 0.7, (255, 255, 255), 2)
    perf = f"{d.fps:4.1f} FPS  {d.infer_ms:5.1f} ms"
    (tw, _), _ = cv2.getTextSize(perf, FONT, 0.5, 1)
    _text(img, perf, (w - tw - 10, 22), 0.5)
    if d.message and d.message_age < 6.0:
        _panel(img, 0, 34, w, 56)
        _text(img, d.message, (10, 50), 0.45, (220, 220, 220))
    if not d.hotkeys_ok:
        _text(img, "Global hotkeys OFF: keep this window focused for Esc", (10, 72), 0.45, (0, 200, 255))

    # Bottom-left gesture panel
    _panel(img, 0, h - 92, 290, h)
    g = d.stable.label or "-"
    _text(img, f"Gesture: {g}  ({d.stable.votes} votes)", (10, h - 70), 0.55)
    y = h - 52
    if d.status.click_progress > 0:
        ctx = "" if d.status.click_context_ok else " (point first)"
        _bar(img, 10, y, 120, "click" + ctx, d.status.click_progress, (0, 140, 255))
        y += 18
    if d.status.right_click_progress > 0:
        _bar(img, 10, y, 120, "right click", d.status.right_click_progress, (0, 210, 210))
        y += 18
    if d.status.toggle_progress > 0:
        _bar(img, 10, y, 120, "disarm/arm", d.status.toggle_progress, (120, 220, 90))
        y += 18
    if d.status.swipe_cooldown:
        _text(img, "swipe cooldown", (10, min(y + 11, h - 6)), 0.42, (180, 180, 180))

    # Last event (fades out)
    if d.last_event and d.last_event_age < 1.5:
        (tw, th), _ = cv2.getTextSize(d.last_event, FONT, 0.9, 2)
        x = (w - tw) // 2
        _panel(img, x - 12, h // 2 - th - 14, x + tw + 12, h // 2 + 12, 0.6)
        _text(img, d.last_event, (x, h // 2), 0.9, (255, 255, 255), 2)

    # Help
    if d.show_help:
        pw = 270
        _panel(img, w - pw, h - 18 * len(HELP_LINES) - 14, w, h)
        for i, line in enumerate(HELP_LINES):
            _text(img, line, (w - pw + 8, h - 18 * (len(HELP_LINES) - i) + 2), 0.42)
    return img
