import threading
import unittest
from dataclasses import replace

from src.config import ALLOWED_KEYS
from src.cursor_controller import ActionExecutor, CursorMapper, FailSafeTriggered, NullBackend
from src.gesture_logic import Action, ActionType
from src.safety import SafetyController, SafetyState
from tests._helpers import FakeClock, default_config


def make_safety(**overrides):
    cfg = default_config()
    clock = FakeClock()
    return SafetyController(replace(cfg.safety, **overrides), clock=clock), clock


class TestStateMachine(unittest.TestCase):
    def test_starts_inactive_and_blocks_everything(self):
        s, _ = make_safety()
        self.assertEqual(s.state, SafetyState.INACTIVE)
        self.assertFalse(s.can_move())
        self.assertFalse(s.permit_discrete())

    def test_arm_grace_period(self):
        s, clock = make_safety(arm_grace_s=0.5)
        self.assertTrue(s.arm())
        self.assertFalse(s.can_move())
        clock.advance(0.6)
        self.assertTrue(s.can_move())

    def test_disarm(self):
        s, clock = make_safety(arm_grace_s=0.0)
        s.arm()
        s.disarm()
        self.assertEqual(s.state, SafetyState.INACTIVE)
        self.assertFalse(s.can_move())

    def test_estop_is_latched(self):
        s, clock = make_safety(arm_grace_s=0.0)
        s.arm()
        s.emergency_stop("test")
        self.assertEqual(s.state, SafetyState.ESTOP)
        self.assertFalse(s.arm())
        s.toggle_arm()
        self.assertEqual(s.state, SafetyState.ESTOP)
        self.assertFalse(s.can_move())

    def test_reset_estop_goes_to_inactive_not_active(self):
        s, _ = make_safety()
        s.emergency_stop()
        self.assertTrue(s.reset_estop())
        self.assertEqual(s.state, SafetyState.INACTIVE)
        self.assertFalse(s.reset_estop())  # nothing to reset now

    def test_estop_from_inactive(self):
        s, _ = make_safety()
        s.emergency_stop()
        self.assertEqual(s.state, SafetyState.ESTOP)

    def test_generation_changes_on_transitions(self):
        s, _ = make_safety()
        g0 = s.generation
        s.arm()
        s.disarm()
        s.emergency_stop()
        self.assertEqual(s.generation, g0 + 3)


class TestGestureToggle(unittest.TestCase):
    def test_gesture_can_disarm(self):
        s, _ = make_safety()
        s.arm()
        s.gesture_toggle()
        self.assertEqual(s.state, SafetyState.INACTIVE)

    def test_gesture_cannot_arm_by_default(self):
        s, _ = make_safety(allow_gesture_arming=False)
        s.gesture_toggle()
        self.assertEqual(s.state, SafetyState.INACTIVE)

    def test_gesture_can_arm_when_enabled(self):
        s, _ = make_safety(allow_gesture_arming=True)
        s.gesture_toggle()
        self.assertEqual(s.state, SafetyState.ACTIVE)

    def test_gesture_cannot_clear_estop(self):
        s, _ = make_safety(allow_gesture_arming=True)
        s.emergency_stop()
        s.gesture_toggle()
        self.assertEqual(s.state, SafetyState.ESTOP)


class TestLimits(unittest.TestCase):
    def test_rate_limit(self):
        s, clock = make_safety(arm_grace_s=0.0, max_discrete_actions_per_s=3)
        s.arm()
        granted = [s.permit_discrete() for _ in range(10)]
        self.assertEqual(sum(granted), 3)
        clock.advance(1.01)
        self.assertTrue(s.permit_discrete())

    def test_idle_auto_disarm(self):
        s, clock = make_safety(arm_grace_s=0.0, idle_disarm_s=10)
        s.arm()
        s.tick(hand_present=True)
        clock.advance(9)
        s.tick(hand_present=False)
        self.assertEqual(s.state, SafetyState.ACTIVE)
        clock.advance(2)
        s.tick(hand_present=False)
        self.assertEqual(s.state, SafetyState.INACTIVE)

    def test_idle_disarm_disabled_with_zero(self):
        s, clock = make_safety(arm_grace_s=0.0, idle_disarm_s=0)
        s.arm()
        clock.advance(10_000)
        s.tick(hand_present=False)
        self.assertEqual(s.state, SafetyState.ACTIVE)

    def test_estop_from_other_thread(self):
        s, _ = make_safety(arm_grace_s=0.0)
        s.arm()
        t = threading.Thread(target=s.emergency_stop, args=("thread",))
        t.start()
        t.join()
        self.assertEqual(s.state, SafetyState.ESTOP)


class _ExplodingBackend(NullBackend):
    def click(self, button):
        raise FailSafeTriggered("corner")


class _BrokenBackend(NullBackend):
    def press(self, key):
        raise OSError("display gone")


class TestExecutorGating(unittest.TestCase):
    def _executor(self, backend=None, **safety_overrides):
        cfg = default_config()
        clock = FakeClock()
        safety = SafetyController(replace(cfg.safety, arm_grace_s=0.0, **safety_overrides), clock=clock)
        backend = backend or NullBackend()
        mapper = CursorMapper(cfg.cursor, *backend.screen_size())
        return ActionExecutor(backend, mapper, safety), safety, backend, clock

    def test_inactive_executes_nothing(self):
        ex, safety, backend, clock = self._executor()
        for a in (Action(ActionType.MOVE, 0.5, 0.5), Action(ActionType.LEFT_CLICK),
                  Action(ActionType.KEY, key="right")):
            ex.execute(a, clock())
        self.assertEqual(backend.calls, [])

    def test_active_executes(self):
        ex, safety, backend, clock = self._executor()
        safety.arm()
        ex.execute(Action(ActionType.LEFT_CLICK), clock())
        ex.execute(Action(ActionType.KEY, key="right"), clock.advance(0.1))
        self.assertEqual(backend.calls, [("click", "left"), ("press", "right")])

    def test_disallowed_key_blocked_even_when_active(self):
        ex, safety, backend, clock = self._executor()
        safety.arm()
        for key in ("enter", "win", "alt", "a", "ctrl"):
            self.assertNotIn(key, ALLOWED_KEYS)
            ex.execute(Action(ActionType.KEY, key=key), clock.advance(1.0))
        self.assertEqual(backend.calls, [])

    def test_failsafe_triggers_estop(self):
        ex, safety, backend, clock = self._executor(_ExplodingBackend())
        safety.arm()
        ex.execute(Action(ActionType.LEFT_CLICK), clock())
        self.assertEqual(safety.state, SafetyState.ESTOP)

    def test_unknown_backend_error_fails_closed(self):
        ex, safety, backend, clock = self._executor(_BrokenBackend())
        safety.arm()
        ex.execute(Action(ActionType.KEY, key="left"), clock())
        self.assertEqual(safety.state, SafetyState.ESTOP)

    def test_estop_blocks_execution(self):
        ex, safety, backend, clock = self._executor()
        safety.arm()
        safety.emergency_stop()
        ex.execute(Action(ActionType.LEFT_CLICK), clock())
        ex.execute(Action(ActionType.MOVE, 0.5, 0.5), clock())
        self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
