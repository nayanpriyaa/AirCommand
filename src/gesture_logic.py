"""Pure gesture logic: hand selection, temporal smoothing, hold triggers, swipes, action emission.

No OpenCV, no Ultralytics, no PyAutoGUI. Every function takes ``now`` explicitly so the
state machine is deterministic and unit-testable with a fake clock.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional, Sequence

from src.config import AppConfig, ActionsConfig, ModelConfig, RolesConfig, SmoothingConfig, SwipeConfig
from src.detector import Detection


class ActionType(str, Enum):
    MOVE = "move"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    KEY = "key"
    TOGGLE_ARM = "toggle_arm"


@dataclass(frozen=True)
class Action:
    type: ActionType
    x: float = 0.0          # normalized frame coords (MOVE only)
    y: float = 0.0
    key: Optional[str] = None  # KEY only; validated against config.ALLOWED_KEYS before execution
    reason: str = ""


@dataclass(frozen=True)
class StableGesture:
    label: Optional[str]
    votes: int
    confidence: float


# --------------------------------------------------------------------------- hand selection


class HandSelector:
    """Pick one primary hand per frame and keep following it across frames."""

    MAX_LOST_FRAMES = 3

    def __init__(self, model_cfg: ModelConfig, smoothing: SmoothingConfig, neutral_label: Optional[str]) -> None:
        self._model_cfg = model_cfg
        self._max_dist = smoothing.track_max_dist
        self._neutral = neutral_label
        self._prev_center: Optional[tuple[float, float]] = None
        self._lost = 0

    def reset(self) -> None:
        self._prev_center = None
        self._lost = 0

    def select(self, detections: Sequence[Detection]) -> Optional[Detection]:
        cands = [d for d in detections if d.confidence >= self._model_cfg.threshold_for(d.label)]
        if not cands:
            self._lost += 1
            if self._lost > self.MAX_LOST_FRAMES:
                self._prev_center = None
            return None
        self._lost = 0
        gesturing = [d for d in cands if d.label != self._neutral] or cands
        chosen: Optional[Detection] = None
        if self._prev_center is not None:
            px, py = self._prev_center
            nearest = min(gesturing, key=lambda d: math.dist(d.center, (px, py)))
            if math.dist(nearest.center, (px, py)) <= self._max_dist:
                chosen = nearest
        if chosen is None:
            chosen = max(gesturing, key=lambda d: (d.confidence, d.area))
        self._prev_center = chosen.center
        return chosen


# --------------------------------------------------------------------------- smoothing


class GestureStabilizer:
    """Majority vote over the last ``window`` frames. Requires ``min_votes`` (a strict majority).

    A missing detection is a vote for "nothing", so a single spurious frame can never become
    the stable gesture.
    """

    def __init__(self, window: int, min_votes: int) -> None:
        if not (window // 2 < min_votes <= window):
            raise ValueError("min_votes must be a strict majority of window")
        self.window = window
        self.min_votes = min_votes
        self._labels: Deque[Optional[str]] = deque(maxlen=window)
        self._confs: Deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self._labels.clear()
        self._confs.clear()

    def update(self, detection: Optional[Detection]) -> StableGesture:
        self._labels.append(detection.label if detection else None)
        self._confs.append(detection.confidence if detection else 0.0)
        counts = Counter(self._labels)
        label, votes = counts.most_common(1)[0]
        if label is None or votes < self.min_votes:
            return StableGesture(None, votes if label is not None else 0, 0.0)
        confs = [c for lab, c in zip(self._labels, self._confs) if lab == label]
        return StableGesture(label, votes, sum(confs) / len(confs))


# --------------------------------------------------------------------------- triggers


class HoldTrigger:
    """Fires once when a condition has been continuously true for ``hold_s``.

    Edge-triggered: after firing, the condition must become false (gesture released) before it
    can fire again. ``cooldown_s`` additionally bounds the firing rate.
    """

    def __init__(self, hold_s: float, cooldown_s: float) -> None:
        self.hold_s = hold_s
        self.cooldown_s = cooldown_s
        self._start: Optional[float] = None
        self._fired = False
        self._last_fire = -math.inf

    def reset(self) -> None:
        self._start = None
        self._fired = False

    def block_until_release(self) -> None:
        """Ignore the current hold (if any); the gesture must be released first."""
        self._fired = True

    def update(self, active: bool, now: float, allowed: bool = True) -> bool:
        if not active:
            self._start = None
            self._fired = False
            return False
        if self._start is None:
            self._start = now
        if self._fired or not allowed:
            return False
        if now - self._start >= self.hold_s and now - self._last_fire >= self.cooldown_s:
            self._fired = True
            self._last_fire = now
            return True
        return False

    def progress(self, now: float) -> float:
        if self._start is None:
            return 0.0
        if self._fired:
            return 1.0
        return max(0.0, min(1.0, (now - self._start) / self.hold_s))


class SwipeDetector:
    """Detects a fast, mostly horizontal, mostly monotonic movement of the swipe-gesture hand."""

    def __init__(self, cfg: SwipeConfig) -> None:
        self.cfg = cfg
        self._points: Deque[tuple[float, float, float]] = deque()
        self._last_fire = -math.inf

    @property
    def trail(self) -> list[tuple[float, float]]:
        return [(x, y) for _, x, y in self._points]

    def reset(self) -> None:
        self._points.clear()

    def in_cooldown(self, now: float) -> bool:
        return now - self._last_fire < self.cfg.cooldown_s

    def _prune(self, now: float) -> None:
        while self._points and now - self._points[0][0] > self.cfg.window_s:
            self._points.popleft()

    def tick(self, now: float) -> None:
        self._prune(now)

    def update(self, now: float, x: float, y: float) -> Optional[str]:
        """Feed one hand position. Returns 'left' or 'right' when a swipe completes."""
        if self.in_cooldown(now):
            return None
        self._points.append((now, x, y))
        self._prune(now)
        if len(self._points) < self.cfg.min_points:
            return None
        _, x0, y0 = self._points[0]
        dx, dy = x - x0, y - y0
        if abs(dx) < self.cfg.min_dx or abs(dy) > self.cfg.max_dy_ratio * abs(dx):
            return None
        sign = 1.0 if dx > 0 else -1.0
        xs = [p[1] for p in self._points]
        steps = [b - a for a, b in zip(xs, xs[1:]) if b != a]
        if not steps or sum(1 for s in steps if s * sign > 0) / len(steps) < self.cfg.monotonic_ratio:
            return None
        self._last_fire = now
        self._points.clear()
        return "right" if dx > 0 else "left"


# --------------------------------------------------------------------------- engine


@dataclass
class EngineStatus:
    click_progress: float = 0.0
    right_click_progress: float = 0.0
    toggle_progress: float = 0.0
    swipe_trail: list[tuple[float, float]] = field(default_factory=list)
    click_context_ok: bool = False
    swipe_cooldown: bool = False


class GestureEngine:
    """Turns (primary detection, stable gesture) per frame into a short list of ``Action``s.

    Guarantees:
      * MOVE only when both the current frame and the smoothed label are the pointer gesture.
      * Each click / right-click / key / toggle fires at most once per continuous hold.
      * Discrete actions need ``min_votes`` agreeing frames AND a minimum hold time.
    """

    def __init__(self, roles: RolesConfig, actions: ActionsConfig, swipe: SwipeConfig, anchor_y: float) -> None:
        self.roles = roles
        self.acfg = actions
        self.scfg = swipe
        self.anchor_y = anchor_y
        self.click = HoldTrigger(actions.click_hold_s, actions.click_cooldown_s)
        self.right_click = HoldTrigger(actions.right_click_hold_s, actions.click_cooldown_s)
        self.toggle = HoldTrigger(actions.toggle_hold_s, actions.toggle_cooldown_s)
        self.swipe = SwipeDetector(swipe)
        self._last_pointer_t = -math.inf
        self._click_was_active = False
        self._click_ctx = False
        self.status = EngineStatus()

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "GestureEngine":
        return cls(cfg.roles, cfg.actions, cfg.swipe, cfg.cursor.anchor_y)

    def reset(self) -> None:
        """Forget all timers. Gestures being held right now must be released before they act."""
        for trig in (self.click, self.right_click, self.toggle):
            trig.reset()
            trig.block_until_release()
        self.swipe.reset()
        self._last_pointer_t = -math.inf
        self._click_was_active = False
        self._click_ctx = False

    def update(self, primary: Optional[Detection], stable: StableGesture, now: float) -> list[Action]:
        roles = self.roles
        raw = primary.label if primary is not None else None
        label = stable.label
        out: list[Action] = []

        # Pointer: absolute cursor positioning from the fingertip estimate.
        if label == roles.pointer:
            self._last_pointer_t = now
            if raw == roles.pointer and primary is not None:
                ax, ay = primary.anchor(self.anchor_y)
                out.append(Action(ActionType.MOVE, ax, ay, reason="pointer"))

        # Left click: fist held, normally only right after pointing (prevents random fists clicking).
        click_active = label == roles.click
        if click_active and not self._click_was_active:
            self._click_ctx = (not self.acfg.click_requires_pointer_context
                               or now - self._last_pointer_t <= self.acfg.pointer_context_s)
        self._click_was_active = click_active
        if self.click.update(click_active, now, allowed=self._click_ctx):
            out.append(Action(ActionType.LEFT_CLICK, reason=f"{roles.click} hold"))

        if roles.right_click and self.right_click.update(label == roles.right_click, now):
            out.append(Action(ActionType.RIGHT_CLICK, reason=f"{roles.right_click} hold"))

        if roles.toggle and self.toggle.update(label == roles.toggle, now):
            out.append(Action(ActionType.TOGGLE_ARM, reason=f"{roles.toggle} hold"))

        # Swipe: uses raw per-frame positions (motion blur makes the smoothed label lag during swipes).
        if self.scfg.enabled and roles.swipe:
            if raw == roles.swipe and primary is not None:
                cx, cy = primary.center
                direction = self.swipe.update(now, cx, cy)
                if direction is not None:
                    key = self.scfg.left_key if direction == "left" else self.scfg.right_key
                    out.append(Action(ActionType.KEY, key=key, reason=f"swipe {direction}"))
            else:
                if raw is not None and raw not in (roles.swipe, roles.neutral):
                    self.swipe.reset()  # a different gesture interrupts the swipe
                self.swipe.tick(now)

        self.status = EngineStatus(
            click_progress=self.click.progress(now) if click_active else 0.0,
            right_click_progress=self.right_click.progress(now) if label == roles.right_click else 0.0,
            toggle_progress=self.toggle.progress(now) if label == roles.toggle else 0.0,
            swipe_trail=self.swipe.trail,
            click_context_ok=self._click_ctx if click_active else
            (now - self._last_pointer_t <= self.acfg.pointer_context_s),
            swipe_cooldown=self.swipe.in_cooldown(now),
        )
        return out
