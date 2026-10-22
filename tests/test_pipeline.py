"""End-to-end pipeline test with a scripted fake detector: no model, camera, or desktop required."""

import unittest

import numpy as np

from src.cursor_controller import NullBackend
from src.main import AirCommandApp
from src.safety import SafetyController, SafetyState
from tests._helpers import FakeClock, default_config, det

DT = 1 / 30


class ScriptedDetector:
    def __init__(self, class_names):
        self.class_names = class_names
        self.next = []

    def detect(self, frame):
        return list(self.next)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.cfg = default_config()
        self.clock = FakeClock()
        self.safety = SafetyController(self.cfg.safety, clock=self.clock)
        self.detector = ScriptedDetector(self.cfg.model.classes)
        self.backend = NullBackend()
        self.app = AirCommandApp(self.cfg, self.detector, self.backend, self.safety)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def feed(self, detections, n):
        events = []
        for _ in range(n):
            self.detector.next = detections
            events += self.app.process(self.frame, self.clock()).events
            self.clock.advance(DT)
        return events

    def calls(self, kind):
        return [c for c in self.backend.calls if c[0] == kind]

    def test_nothing_happens_until_armed(self):
        self.feed([det("one")], 30)
        self.feed([det("fist")], 30)
        self.feed([det("palm", cx=0.7)], 5)
        self.assertEqual(self.backend.calls, [])

    def test_point_click_flow_when_armed(self):
        self.safety.arm(now=self.clock())
        self.feed([det("one", cx=0.5, cy=0.5)], 40)  # beyond the arm grace period
        self.assertGreater(len(self.calls("move")), 0)
        self.feed([det("fist")], 30)
        self.assertEqual(self.calls("click"), [("click", "left")])

    def test_arming_while_holding_gesture_does_not_fire(self):
        self.feed([det("peace")], 10)
        self.safety.arm(now=self.clock())
        self.feed([det("peace")], 90)
        self.assertEqual(self.calls("click"), [])

    def test_thumbs_up_disarms(self):
        self.safety.arm(now=self.clock())
        self.feed([det("like")], 70)
        self.assertEqual(self.safety.state, SafetyState.INACTIVE)

    def test_swipe_presses_key_when_armed(self):
        self.safety.arm(now=self.clock())
        self.feed([det("palm", cx=0.7)], 30)
        for i in range(8):
            self.feed([det("palm", cx=0.7 - i * 0.06)], 1)
        self.assertEqual(self.calls("press"), [("press", self.cfg.swipe.left_key)])

    def test_estop_mid_session_blocks_actions(self):
        self.safety.arm(now=self.clock())
        self.feed([det("one")], 40)
        self.safety.emergency_stop("test", now=self.clock())
        before = len(self.backend.calls)
        self.feed([det("one", cx=0.3)], 20)
        self.feed([det("fist")], 30)
        self.assertEqual(len(self.backend.calls), before)


if __name__ == "__main__":
    unittest.main()
