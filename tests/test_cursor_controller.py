import unittest
from dataclasses import replace

from src.cursor_controller import CursorMapper, OneEuroFilter
from tests._helpers import default_config


class TestOneEuro(unittest.TestCase):
    def test_converges_to_constant_input(self):
        f = OneEuroFilter(1.0, 0.01, 1.0)
        out = f(0.0, 0.0)
        for i in range(1, 200):
            out = f(100.0, i / 30)
        self.assertAlmostEqual(out, 100.0, delta=0.5)

    def test_smooths_jitter(self):
        f = OneEuroFilter(1.0, 0.0, 1.0)
        outs = [f(100.0 + (5 if i % 2 else -5), i / 30) for i in range(60)]
        spread = max(outs[30:]) - min(outs[30:])
        self.assertLess(spread, 10.0)  # raw spread is 10

    def test_ignores_non_increasing_timestamps(self):
        f = OneEuroFilter(1.0, 0.0, 1.0)
        f(10.0, 1.0)
        self.assertEqual(f(50.0, 1.0), 10.0)


class TestMapper(unittest.TestCase):
    def setUp(self):
        self.cfg = default_config().cursor

    def test_region_corners_map_inside_margins(self):
        m = CursorMapper(self.cfg, 1920, 1080)
        x1, y1, x2, y2 = m.effective_region
        tl = m.to_screen(x1, y1)
        br = m.to_screen(x2, y2)
        self.assertEqual(tl, (self.cfg.edge_margin_px, self.cfg.edge_margin_px))
        self.assertEqual(br, (1919 - self.cfg.edge_margin_px, 1079 - self.cfg.edge_margin_px))

    def test_outside_region_is_clamped_never_reaches_corner(self):
        m = CursorMapper(self.cfg, 1920, 1080)
        for nx, ny in ((0, 0), (1, 1), (0, 1), (1, 0)):
            m.reset()
            px, py = m.map(nx, ny, 0.0)
            self.assertTrue(self.cfg.edge_margin_px <= px <= 1919 - self.cfg.edge_margin_px)
            self.assertTrue(self.cfg.edge_margin_px <= py <= 1079 - self.cfg.edge_margin_px)

    def test_sensitivity_shrinks_region(self):
        base = CursorMapper(self.cfg, 1920, 1080).effective_region
        fast = CursorMapper(replace(self.cfg, sensitivity=2.0), 1920, 1080).effective_region
        self.assertLess(fast[2] - fast[0], base[2] - base[0])

    def test_deadzone_suppresses_tiny_moves(self):
        m = CursorMapper(replace(self.cfg, deadzone_px=10), 1920, 1080)
        self.assertIsNotNone(m.map(0.5, 0.45, 0.0))
        self.assertIsNone(m.map(0.5001, 0.45, 1 / 30))

    def test_gap_resets_filter_to_jump(self):
        m = CursorMapper(self.cfg, 1920, 1080)
        m.map(0.25, 0.2, 0.0)
        far = m.map(0.75, 0.7, 5.0)  # long pause: jump directly instead of gliding
        self.assertEqual(far, tuple(int(round(v)) for v in m.to_screen(0.75, 0.7)))


if __name__ == "__main__":
    unittest.main()
