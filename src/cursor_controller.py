"""Cursor smoothing/mapping and the only code that touches the desktop (via a swappable backend)."""

from __future__ import annotations

import math
import platform
from dataclasses import dataclass, field
from typing import Optional, Protocol

from src.config import ALLOWED_KEYS, CursorConfig
from src.gesture_logic import Action, ActionType
from src.safety import SafetyController


class DesktopError(RuntimeError):
    """Desktop automation backend could not be initialised."""


class FailSafeTriggered(RuntimeError):
    """The user moved the physical mouse into a screen corner (PyAutoGUI fail-safe)."""


# --------------------------------------------------------------------------- filtering


class OneEuroFilter:
    """One Euro filter (Casiez et al., CHI 2012): adaptive low-pass, smooth at rest, responsive when fast."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: Optional[float] = None
        self._dx = 0.0
        self._t: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x, self._dx, self._t = None, 0.0, None

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None:
            self._x, self._t = x, t
            return x
        if t <= self._t:  # duplicate/out-of-order timestamp: keep the last estimate
            return self._x
        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        self._t = t
        return self._x


class CursorMapper:
    """Maps a normalized frame point to a smoothed, clamped screen pixel."""

    GAP_RESET_S = 0.35  # re-entering pointer mode after this gap jumps instead of gliding

    def __init__(self, cfg: CursorConfig, screen_w: int, screen_h: int) -> None:
        self.cfg = cfg
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._fx = OneEuroFilter(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
        self._fy = OneEuroFilter(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
        self._last_out: Optional[tuple[int, int]] = None
        self._last_t: Optional[float] = None

    @property
    def effective_region(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.cfg.active_region
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw, hh = (x2 - x1) / 2 / self.cfg.sensitivity, (y2 - y1) / 2 / self.cfg.sensitivity
        return (max(0.0, cx - hw), max(0.0, cy - hh), min(1.0, cx + hw), min(1.0, cy + hh))

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()
        self._last_out = None
        self._last_t = None

    def to_screen(self, nx: float, ny: float) -> tuple[float, float]:
        x1, y1, x2, y2 = self.effective_region
        u = min(1.0, max(0.0, (nx - x1) / (x2 - x1)))
        v = min(1.0, max(0.0, (ny - y1) / (y2 - y1)))
        m = self.cfg.edge_margin_px
        return (m + u * (self.screen_w - 1 - 2 * m), m + v * (self.screen_h - 1 - 2 * m))

    def map(self, nx: float, ny: float, now: float) -> Optional[tuple[int, int]]:
        """Return a new pixel target, or None if the movement is inside the dead zone."""
        if self._last_t is not None and now - self._last_t > self.GAP_RESET_S:
            self.reset()
        self._last_t = now
        sx, sy = self.to_screen(nx, ny)
        fx, fy = self._fx(sx, now), self._fy(sy, now)
        m = self.cfg.edge_margin_px
        px = int(round(min(self.screen_w - 1 - m, max(m, fx))))
        py = int(round(min(self.screen_h - 1 - m, max(m, fy))))
        if self._last_out is not None and math.dist(self._last_out, (px, py)) < self.cfg.deadzone_px:
            return None
        self._last_out = (px, py)
        return (px, py)


# --------------------------------------------------------------------------- backends


class DesktopBackend(Protocol):
    def screen_size(self) -> tuple[int, int]: ...
    def move_to(self, x: int, y: int) -> None: ...
    def click(self, button: str) -> None: ...
    def press(self, key: str) -> None: ...


@dataclass
class NullBackend:
    """Records calls instead of acting. Used for --dry-run and tests."""

    size: tuple[int, int] = (1920, 1080)
    calls: list[tuple] = field(default_factory=list)
    verbose: bool = False

    def screen_size(self) -> tuple[int, int]:
        return self.size

    def move_to(self, x: int, y: int) -> None:
        self.calls.append(("move", x, y))

    def click(self, button: str) -> None:
        self.calls.append(("click", button))
        if self.verbose:
            print(f"[dry-run] {button} click")

    def press(self, key: str) -> None:
        self.calls.append(("press", key))
        if self.verbose:
            print(f"[dry-run] key {key}")


class PyAutoGuiBackend:
    """Real desktop control through PyAutoGUI."""

    def __init__(self, failsafe: bool = True) -> None:
        try:
            import pyautogui
        except Exception as exc:  # KeyError: 'DISPLAY', Xlib errors, missing tkinter, ...
            hint = ""
            if platform.system() == "Linux":
                hint = ("\nLinux: run inside an X11 session (not pure Wayland), ensure $DISPLAY is set, and "
                        "install: sudo apt install python3-tk python3-dev")
            raise DesktopError(f"PyAutoGUI could not start: {exc!r}{hint}\n"
                               "Use --dry-run to run without desktop control.") from exc
        self._pg = pyautogui
        pyautogui.FAILSAFE = failsafe
        pyautogui.PAUSE = 0.0  # default 0.1 s sleep after every call would cap the loop at ~10 FPS

    def screen_size(self) -> tuple[int, int]:
        w, h = self._pg.size()
        return int(w), int(h)

    def _guard(self, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except self._pg.FailSafeException as exc:
            raise FailSafeTriggered(str(exc)) from exc

    def move_to(self, x: int, y: int) -> None:
        self._guard(self._pg.moveTo, x, y, _pause=False)

    def click(self, button: str) -> None:
        self._guard(self._pg.click, button=button, _pause=False)

    def press(self, key: str) -> None:
        self._guard(self._pg.press, key, _pause=False)


# --------------------------------------------------------------------------- executor


class ActionExecutor:
    """Gatekeeper between gesture actions and the desktop. Fails closed on any backend error."""

    def __init__(self, backend: DesktopBackend, mapper: CursorMapper, safety: SafetyController) -> None:
        self.backend = backend
        self.mapper = mapper
        self.safety = safety
        self.counts: dict[str, int] = {}
        self.last_event: str = ""
        self.last_cursor: Optional[tuple[int, int]] = None

    def _count(self, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1

    def execute(self, action: Action, now: float) -> Optional[str]:
        """Run one action if safety allows. Returns a short description of what happened, else None."""
        try:
            if action.type == ActionType.MOVE:
                if not self.safety.can_move(now):
                    return None
                target = self.mapper.map(action.x, action.y, now)
                if target is None:
                    return None
                self.backend.move_to(*target)
                self.last_cursor = target
                self._count("move")
                return None  # moves are too frequent to report
            if action.type == ActionType.TOGGLE_ARM:
                return None  # handled by the caller via SafetyController.gesture_toggle
            if action.type == ActionType.KEY and action.key not in ALLOWED_KEYS:
                self.last_event = f"blocked key {action.key!r}"
                return self.last_event
            if not self.safety.permit_discrete(now):
                return None
            if action.type == ActionType.LEFT_CLICK:
                self.backend.click("left")
                desc = "LEFT CLICK"
            elif action.type == ActionType.RIGHT_CLICK:
                self.backend.click("right")
                desc = "RIGHT CLICK"
            elif action.type == ActionType.KEY:
                self.backend.press(action.key)  # type: ignore[arg-type]
                desc = f"KEY {action.key.upper()} ({action.reason})"  # type: ignore[union-attr]
            else:
                return None
            self._count(action.type.value)
            self.last_event = desc
            return desc
        except FailSafeTriggered:
            self.safety.emergency_stop("mouse moved to screen corner", now)
            self.last_event = "FAIL-SAFE corner"
            return self.last_event
        except Exception as exc:  # unknown backend failure: stop everything rather than guess
            self.safety.emergency_stop(f"backend error: {exc}", now)
            self.last_event = f"backend error: {exc}"
            return self.last_event
