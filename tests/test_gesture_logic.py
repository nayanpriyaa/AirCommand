import unittest

from src.gesture_logic import (ActionType, GestureEngine, GestureStabilizer, HandSelector, HoldTrigger,
                               StableGesture, SwipeDetector)
from tests._helpers import default_config, det

DT = 1 / 30  # simulated 30 FPS


def run_engine(engine: GestureEngine, stabilizer: GestureStabilizer, frames, t0: float = 100.0):
    """frames: list of Detection | None. Returns (actions per frame, final time)."""
    t = t0
    out = []
    for d in frames:
        stable = stabilizer.update(d)
        out.append(engine.update(d, stable, t))
        t += DT
    return out, t


def flat(actions_per_frame, kind=None):
    acts = [a for frame in actions_per_frame for a in frame]
    return [a for a in acts if kind is None or a.type == kind]


class TestStabilizer(unittest.TestCase):
    def test_single_spurious_frame_never_becomes_stable(self):
        s = GestureStabilizer(5, 3)
        seq = [det("one")] * 5 + [det("fist")] + [det("one")] * 5
        labels = [s.update(d).label for d in seq]
        self.assertNotIn("fist", labels)

    def test_requires_min_votes(self):
        s = GestureStabilizer(5, 3)
        self.assertIsNone(s.update(det("palm")).label)
        self.assertIsNone(s.update(det("palm")).label)
        self.assertEqual(s.update(det("palm")).label, "palm")

    def test_missing_detections_vote_for_nothing(self):
        s = GestureStabilizer(5, 3)
        for _ in range(5):
            s.update(det("one"))
        for _ in range(2):
            self.assertEqual(s.update(None).label, "one")
        self.assertIsNone(s.update(None).label)

    def test_rejects_non_majority(self):
        with self.assertRaises(ValueError):
            GestureStabilizer(6, 3)


class TestHandSelector(unittest.TestCase):
    def setUp(self):
        cfg = default_config()
        self.sel = HandSelector(cfg.model, cfg.smoothing, cfg.roles.neutral)

    def test_prefers_gesture_over_neutral(self):
        chosen = self.sel.select([det("no_gesture", 0.95, 0.3), det("one", 0.7, 0.7)])
        self.assertEqual(chosen.label, "one")

    def test_applies_per_class_threshold(self):
        self.assertIsNone(self.sel.select([det("fist", 0.58)]))  # fist threshold is 0.60
        self.assertEqual(self.sel.select([det("fist", 0.61)]).label, "fist")

    def test_tracks_nearest_hand(self):
        self.sel.select([det("one", 0.9, 0.3, 0.5)])
        chosen = self.sel.select([det("one", 0.95, 0.8, 0.5), det("one", 0.7, 0.32, 0.5)])
        self.assertAlmostEqual(chosen.center[0], 0.32)


class TestHoldTrigger(unittest.TestCase):
    def test_fires_once_per_hold_and_rearms_after_release(self):
        h = HoldTrigger(0.2, 0.5)
        fired = [h.update(True, t / 100) for t in range(0, 100)]
        self.assertEqual(sum(fired), 1)
        h.update(False, 1.01)
        fired = [h.update(True, 1.02 + t / 100) for t in range(0, 50)]
        self.assertEqual(sum(fired), 1)

    def test_cooldown(self):
        h = HoldTrigger(0.0, 1.0)
        self.assertTrue(h.update(True, 0.0))
        h.update(False, 0.1)
        self.assertFalse(h.update(True, 0.2))  # still cooling down

    def test_block_until_release(self):
        h = HoldTrigger(0.1, 0.1)
        h.update(True, 0.0)
        h.block_until_release()
        self.assertFalse(any(h.update(True, t / 10) for t in range(1, 30)))
        h.update(False, 3.1)
        self.assertTrue(any(h.update(True, 3.2 + t / 10) for t in range(0, 5)))


class TestClickLogic(unittest.TestCase):
    def setUp(self):
        self.cfg = default_config()
        self.engine = GestureEngine.from_config(self.cfg)
        self.stab = GestureStabilizer(self.cfg.smoothing.window, self.cfg.smoothing.min_votes)

    def test_point_then_fist_clicks_exactly_once(self):
        frames = [det("one")] * 15 + [det("fist")] * 60
        acts, _ = run_engine(self.engine, self.stab, frames)
        self.assertEqual(len(flat(acts, ActionType.LEFT_CLICK)), 1)

    def test_single_fist_frame_never_clicks(self):
        frames = [det("one")] * 15 + [det("fist")] + [det("one")] * 15
        acts, _ = run_engine(self.engine, self.stab, frames)
        self.assertEqual(flat(acts, ActionType.LEFT_CLICK), [])

    def test_short_fist_below_hold_time_does_not_click(self):
        n = int(self.cfg.actions.click_hold_s / DT)  # stable only for less than the hold time
        frames = [det("one")] * 15 + [det("fist")] * n + [det("one")] * 15
        acts, _ = run_engine(self.engine, self.stab, frames)
        self.assertEqual(flat(acts, ActionType.LEFT_CLICK), [])

    def test_fist_without_recent_pointer_does_not_click(self):
        frames = [None] * 60 + [det("fist")] * 60
        acts, _ = run_engine(self.engine, self.stab, frames)
        self.assertEqual(flat(acts, ActionType.LEFT_CLICK), [])

    def test_two_separate_clicks_need_release(self):
        frames = ([det("one")] * 15 + [det("fist")] * 30) * 2
        acts, _ = run_engine(self.engine, self.stab, frames)
        self.assertEqual(len(flat(acts, ActionType.LEFT_CLICK)), 2)

    def test_right_click_once_per_hold(self):
        acts, _ = run_engine(self.engine, self.stab, [det("peace")] * 90)
        self.assertEqual(len(flat(acts, ActionType.RIGHT_CLICK)), 1)


class TestPointer(unittest.TestCase):
    def test_move_only_when_raw_and_stable_agree(self):
        cfg = default_config()
        engine = GestureEngine.from_config(cfg)
        stab = GestureStabilizer(cfg.smoothing.window, cfg.smoothing.min_votes)
        frames = [det("one")] * 10 + [det("fist")]
        acts, _ = run_engine(engine, stab, frames)
        self.assertEqual(acts[-1], [])  # stable is still "one" but this frame is a fist: cursor frozen
        self.assertTrue(all(a.type == ActionType.MOVE for a in flat(acts)))
        self.assertGreater(len(flat(acts, ActionType.MOVE)), 0)


class TestSwipe(unittest.TestCase):
    def setUp(self):
        self.cfg = default_config()

    def _sweep(self, x0, x1, n, y0=0.5, y1=0.5, label="palm"):
        return [det(label, 0.9, x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1)) for i in range(n)]

    def _run(self, frames):
        engine = GestureEngine.from_config(self.cfg)
        stab = GestureStabilizer(self.cfg.smoothing.window, self.cfg.smoothing.min_votes)
        acts, _ = run_engine(engine, stab, frames)
        return flat(acts, ActionType.KEY)

    def test_fast_left_swipe_presses_next(self):
        keys = self._run([det("palm", cx=0.7)] * 10 + self._sweep(0.7, 0.3, 8))
        self.assertEqual([k.key for k in keys], [self.cfg.swipe.left_key])

    def test_fast_right_swipe_presses_previous(self):
        keys = self._run([det("palm", cx=0.3)] * 10 + self._sweep(0.3, 0.7, 8))
        self.assertEqual([k.key for k in keys], [self.cfg.swipe.right_key])

    def test_slow_drift_is_ignored(self):
        self.assertEqual(self._run(self._sweep(0.7, 0.3, 120)), [])

    def test_vertical_motion_is_ignored(self):
        self.assertEqual(self._run(self._sweep(0.5, 0.6, 8, 0.2, 0.8)), [])

    def test_static_palm_is_ignored(self):
        self.assertEqual(self._run([det("palm")] * 90), [])

    def test_back_and_forth_triggers_once_due_to_cooldown(self):
        frames = self._sweep(0.7, 0.3, 8) + self._sweep(0.3, 0.7, 8)
        self.assertEqual(len(self._run(frames)), 1)

    def test_wrong_gesture_does_not_swipe(self):
        self.assertEqual(self._run(self._sweep(0.7, 0.3, 8, label="one")), [])

    def test_detector_unit(self):
        sd = SwipeDetector(self.cfg.swipe)
        results = [sd.update(i * DT, 0.8 - i * 0.06, 0.5) for i in range(8)]
        self.assertEqual([r for r in results if r], ["left"])


class TestToggleAndReset(unittest.TestCase):
    def test_toggle_hold_emits_once(self):
        cfg = default_config()
        engine = GestureEngine.from_config(cfg)
        stab = GestureStabilizer(cfg.smoothing.window, cfg.smoothing.min_votes)
        acts, _ = run_engine(engine, stab, [det("like")] * 150)
        self.assertEqual(len(flat(acts, ActionType.TOGGLE_ARM)), 1)

    def test_reset_requires_release_of_held_gesture(self):
        cfg = default_config()
        engine = GestureEngine.from_config(cfg)
        stab = GestureStabilizer(cfg.smoothing.window, cfg.smoothing.min_votes)
        run_engine(engine, stab, [det("peace")] * 5)
        engine.reset()
        acts, _ = run_engine(engine, stab, [det("peace")] * 90, t0=200.0)
        self.assertEqual(flat(acts, ActionType.RIGHT_CLICK), [])

    def test_neutral_gesture_produces_no_actions(self):
        cfg = default_config()
        engine = GestureEngine.from_config(cfg)
        stab = GestureStabilizer(cfg.smoothing.window, cfg.smoothing.min_votes)
        acts, _ = run_engine(engine, stab, [det("no_gesture")] * 120)
        self.assertEqual(flat(acts), [])

    def test_stable_gesture_dataclass(self):
        self.assertIsNone(StableGesture(None, 0, 0.0).label)


if __name__ == "__main__":
    unittest.main()
