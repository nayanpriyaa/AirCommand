import copy
import unittest

import yaml

from src.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, ConfigError, apply_overrides, load_config, parse_config


def raw_default():
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


class TestConfig(unittest.TestCase):
    def test_default_config_loads(self):
        cfg = load_config()
        self.assertEqual(cfg.model.path, PROJECT_ROOT / "models" / "best.pt")
        self.assertIn(cfg.roles.pointer, cfg.model.classes)
        self.assertEqual(cfg.model.threshold_for("fist"), 0.60)
        self.assertEqual(cfg.model.threshold_for("one"), cfg.model.confidence)

    def _expect_error(self, mutate, fragment):
        raw = copy.deepcopy(raw_default())
        mutate(raw)
        with self.assertRaises(ConfigError) as ctx:
            parse_config(raw)
        self.assertIn(fragment, str(ctx.exception))

    def test_unknown_key_rejected(self):
        self._expect_error(lambda r: r["camera"].update(indx=1), "Unknown key")

    def test_non_majority_votes_rejected(self):
        self._expect_error(lambda r: r["smoothing"].update(window=6, min_votes=3), "strict majority")

    def test_role_must_be_a_class(self):
        self._expect_error(lambda r: r["roles"].update(pointer="pinch"), "not in model.classes")

    def test_duplicate_roles_rejected(self):
        self._expect_error(lambda r: r["roles"].update(click="one"), "only one role")

    def test_disallowed_swipe_key_rejected(self):
        self._expect_error(lambda r: r["swipe"].update(left_key="enter"), "not an allowed key")

    def test_bad_region_rejected(self):
        self._expect_error(lambda r: r["cursor"].update(active_region=[0.8, 0.1, 0.2, 0.9]), "active_region")

    def test_imgsz_multiple_of_32(self):
        self._expect_error(lambda r: r["model"].update(imgsz=500), "multiple of 32")

    def test_overrides(self):
        cfg = apply_overrides(load_config(), camera_index=2, confidence=0.7, model_path="models/other.pt")
        self.assertEqual(cfg.camera.index, 2)
        self.assertEqual(cfg.model.confidence, 0.7)
        self.assertEqual(cfg.model.path.name, "other.pt")
        with self.assertRaises(ConfigError):
            apply_overrides(cfg, confidence=1.5)

    def test_missing_file(self):
        with self.assertRaises(ConfigError):
            load_config("config/does_not_exist.yaml")


if __name__ == "__main__":
    unittest.main()
