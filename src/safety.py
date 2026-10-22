"""Arm/disarm state machine, latched emergency stop, rate limiting, and idle auto-disarm.

Thread-safe: the global hotkey listener calls ``emergency_stop`` from its own thread.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from enum import Enum
from typing import Callable, Deque, Optional

from src.config import SafetyConfig


class SafetyState(str, Enum):
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    ESTOP = "E-STOP"


class SafetyController:
    """Single source of truth for whether desktop actions may run.

    * Starts INACTIVE. There is intentionally no option to start ACTIVE.
    * ESTOP is latched: arming is refused until ``reset_estop`` is called explicitly.
    * A grace period after arming ignores gestures that were already in progress.
    * Discrete actions (clicks, keys) are capped per second regardless of gesture logic.
    """

    def __init__(self, cfg: SafetyConfig, clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self._clock = clock
        self._lock = threading.RLock()
        self._state = SafetyState.INACTIVE
        self._armed_at = -1e18
        self._last_hand_seen = clock()
        self._recent: Deque[float] = deque()
        self._generation = 0
        self._message = "Inactive. Press F8 (or 'a' in the window) to arm."
        self._message_t = clock()

    # ------------------------------------------------------------------ properties
    @property
    def state(self) -> SafetyState:
        with self._lock:
            return self._state

    @property
    def generation(self) -> int:
        """Incremented on every state change; consumers reset their gesture timers when it changes."""
        with self._lock:
            return self._generation

    @property
    def message(self) -> tuple[str, float]:
        with self._lock:
            return self._message, self._message_t

    def _now(self, now: Optional[float]) -> float:
        return self._clock() if now is None else now

    def _set(self, state: SafetyState, msg: str, now: float) -> None:
        if state != self._state:
            self._generation += 1
        self._state = state
        self._message = msg
        self._message_t = now

    # ------------------------------------------------------------------ transitions
    def arm(self, now: Optional[float] = None, source: str = "key") -> bool:
        with self._lock:
            now = self._now(now)
            if self._state == SafetyState.ESTOP:
                self._message, self._message_t = "E-STOP latched. Press F9 (or 'r') to reset first.", now
                return False
            if self._state == SafetyState.ACTIVE:
                return True
            self._armed_at = now
            self._last_hand_seen = now
            self._recent.clear()
            self._set(SafetyState.ACTIVE, f"ARMED ({source}). Desktop control is live.", now)
            return True

    def disarm(self, reason: str = "manual", now: Optional[float] = None) -> None:
        with self._lock:
            now = self._now(now)
            if self._state == SafetyState.ACTIVE:
                self._set(SafetyState.INACTIVE, f"Disarmed ({reason}).", now)

    def toggle_arm(self, now: Optional[float] = None, source: str = "key") -> SafetyState:
        with self._lock:
            if self._state == SafetyState.ACTIVE:
                self.disarm(source, now)
            else:
                self.arm(now, source)
            return self._state

    def emergency_stop(self, reason: str = "Esc", now: Optional[float] = None) -> None:
        with self._lock:
            now = self._now(now)
            self._set(SafetyState.ESTOP, f"EMERGENCY STOP ({reason}). All actions blocked.", now)

    def reset_estop(self, now: Optional[float] = None) -> bool:
        with self._lock:
            now = self._now(now)
            if self._state != SafetyState.ESTOP:
                return False
            self._set(SafetyState.INACTIVE, "E-STOP cleared. Still inactive; arm explicitly.", now)
            return True

    def gesture_toggle(self, now: Optional[float] = None) -> None:
        """Toggle requested by gesture: always allowed to disarm, arms only if configured."""
        with self._lock:
            now = self._now(now)
            if self._state == SafetyState.ACTIVE:
                self.disarm("gesture", now)
            elif self._state == SafetyState.INACTIVE and self.cfg.allow_gesture_arming:
                self.arm(now, "gesture")
            elif self._state == SafetyState.INACTIVE:
                self._message, self._message_t = "Gesture arming disabled; press F8 or 'a'.", now

    # ------------------------------------------------------------------ gating
    def can_move(self, now: Optional[float] = None) -> bool:
        with self._lock:
            now = self._now(now)
            return self._state == SafetyState.ACTIVE and now - self._armed_at >= self.cfg.arm_grace_s

    def permit_discrete(self, now: Optional[float] = None) -> bool:
        """Check and consume one slot of the discrete-action budget."""
        with self._lock:
            now = self._now(now)
            if not self.can_move(now):
                return False
            while self._recent and now - self._recent[0] >= 1.0:
                self._recent.popleft()
            if len(self._recent) >= self.cfg.max_discrete_actions_per_s:
                self._message, self._message_t = "Rate limit hit; action dropped.", now
                return False
            self._recent.append(now)
            return True

    def tick(self, hand_present: bool, now: Optional[float] = None) -> None:
        """Call once per frame. Handles idle auto-disarm."""
        with self._lock:
            now = self._now(now)
            if hand_present:
                self._last_hand_seen = now
            elif (self._state == SafetyState.ACTIVE and self.cfg.idle_disarm_s > 0
                  and now - self._last_hand_seen >= self.cfg.idle_disarm_s):
                self.disarm(f"no hand for {self.cfg.idle_disarm_s:.0f}s", now)


class GlobalHotkeys:
    """System-wide Esc (e-stop), F8 (arm/disarm), F9 (reset e-stop) using pynput.

    Needed because cv2.waitKey only sees keys while the preview window has focus, which it
    loses as soon as you click into another application. Listener is non-suppressing: the key
    still reaches the focused application (e.g. Esc also exits a PowerPoint slideshow).
    """

    DEBOUNCE_S = 0.6

    def __init__(self, safety: SafetyController) -> None:
        self.safety = safety
        self._listener = None
        self.error: Optional[str] = None

    def start(self) -> bool:
        try:
            from pynput import keyboard
        except Exception as exc:  # ImportError, or backend failure (no X server / Wayland-only)
            self.error = f"Global hotkeys unavailable ({exc}). Esc works only while the preview window is focused."
            return False

        last_press: dict[object, float] = {}

        def on_press(key: object) -> None:
            try:
                if key == keyboard.Key.esc:  # idempotent: never debounced
                    self.safety.emergency_stop("Esc")
                    return
                if key not in (keyboard.Key.f8, keyboard.Key.f9):
                    return
                # Debounce: keyboard auto-repeat (and some X11 setups) deliver repeated presses,
                # which would toggle F8 several times for a single tap.
                now = time.monotonic()
                if now - last_press.get(key, -1e9) < self.DEBOUNCE_S:
                    return
                last_press[key] = now
                if key == keyboard.Key.f8:
                    self.safety.toggle_arm(source="F8")
                else:
                    self.safety.reset_estop()
            except Exception:  # never let a callback exception kill the listener thread
                pass

        try:
            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.daemon = True
            self._listener.start()
        except Exception as exc:
            self.error = f"Global hotkeys failed to start ({exc}). Esc works only in the preview window."
            self._listener = None
            return False
        return True

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
