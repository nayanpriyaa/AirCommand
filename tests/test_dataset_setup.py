"""Dataset conversion tests on a synthetic HaGRID-shaped archive (no network)."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from training import dataset_setup as ds

CLASSES = ("one", "fist", "no_gesture")


def make_archives(root: Path, overlap_user: bool = False) -> tuple[Path, Path]:
    ann_zip, img_zip = root / "annotations.zip", root / "images.zip"
    records = {split: {c: {} for c in CLASSES} for split in ds.SPLITS}
    n = 0
    for split in ds.SPLITS:
        for cls in CLASSES:
            for i in range(6):
                stem = f"{split}-{cls}-{i:04d}"
                user = f"user-{split}-{i % 3}"
                if overlap_user and split == "val" and i == 0:
                    user = "user-train-0"  # deliberately leak a train subject into val
                labels = [cls] if cls == "no_gesture" else [cls, "no_gesture"]
                bboxes = [[0.1, 0.2, 0.3, 0.4], [0.8, 0.9, 0.4, 0.4]][: len(labels)]
                records[split][cls][stem] = {"bboxes": bboxes, "labels": labels, "user_id": user,
                                             "united_bbox": None, "united_label": None}
                n += 1
        # One record with a foreign label that must be skipped.
        records[split]["one"][f"{split}-foreign"] = {"bboxes": [[0.1, 0.1, 0.2, 0.2]], "labels": ["call"],
                                                     "user_id": "x"}
    with zipfile.ZipFile(ann_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for split in ds.SPLITS:
            for cls in CLASSES:
                zf.writestr(f"annotations/{split}/{cls}.json", json.dumps(records[split][cls]))
    with zipfile.ZipFile(img_zip, "w") as zf:
        for split in ds.SPLITS:
            for cls in CLASSES:
                for i, stem in enumerate(records[split][cls]):
                    method = zipfile.ZIP_STORED if i % 2 else zipfile.ZIP_DEFLATED
                    zf.writestr(zipfile.ZipInfo(f"hagridv2_512/{cls}/{stem}.jpg"), f"jpeg-{stem}".encode() * 50,
                                compress_type=method)
    return ann_zip, img_zip


class TestConversion(unittest.TestCase):
    def test_bbox_conversion_and_clipping(self):
        lines = ds.hagrid_to_yolo_lines({"bboxes": [[0.1, 0.2, 0.3, 0.4], [0.8, 0.9, 0.4, 0.4]],
                                         "labels": ["one", "no_gesture"]}, {"one": 0, "no_gesture": 2})
        self.assertEqual(lines[0], "0 0.250000 0.400000 0.300000 0.400000")
        cls, cx, cy, w, h = lines[1].split()
        self.assertEqual(cls, "2")
        self.assertAlmostEqual(float(cx) + float(w) / 2, 1.0, places=5)  # clipped at the right edge
        self.assertAlmostEqual(float(cy) + float(h) / 2, 1.0, places=5)

    def test_foreign_label_rejects_image(self):
        self.assertIsNone(ds.hagrid_to_yolo_lines({"bboxes": [[0, 0, 0.1, 0.1]], "labels": ["call"]}, {"one": 0}))

    def test_mismatched_lengths_rejected(self):
        self.assertIsNone(ds.hagrid_to_yolo_lines({"bboxes": [[0, 0, 0.1, 0.1]], "labels": []}, {"one": 0}))


class TestEndToEnd(unittest.TestCase):
    def _run(self, overlap_user=False, quotas=(4, 2, 2)):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        ann, img = make_archives(root, overlap_user)
        out = root / "out"
        code = ds.main(["--out", str(out), "--classes", *CLASSES, "--images-zip", str(img), "--annotations-zip",
                        str(ann), "--train-per-class", str(quotas[0]), "--val-per-class", str(quotas[1]),
                        "--test-per-class", str(quotas[2]), "--workers", "4"])
        return code, out

    def test_builds_yolo_dataset(self):
        code, out = self._run()
        self.assertEqual(code, 0)
        report = json.loads((out / "split_report.json").read_text())
        self.assertEqual(report["splits"]["train"]["images"], 12)
        self.assertEqual(report["splits"]["val"]["images"], 6)
        self.assertEqual(report["selection"]["skipped_foreign_label"] >= 0, True)
        data_yaml = (out / "data.yaml").read_text()
        self.assertIn("0: one", data_yaml)
        self.assertIn("test: images/test", data_yaml)
        img = next((out / "images" / "train").glob("one__*.jpg"))
        self.assertTrue(img.read_bytes().startswith(b"jpeg-train-one"))
        label = (out / "labels" / "train" / (img.stem + ".txt")).read_text().split("\n")[0]
        self.assertTrue(label.startswith("0 "))

    def test_rerun_is_cached_and_shrinking_quota_removes_orphans(self):
        code, out = self._run()
        self.assertEqual(code, 0)
        code2, out2 = self._run(quotas=(2, 1, 1))
        report = json.loads((out2 / "split_report.json").read_text())
        self.assertEqual(report["splits"]["train"]["images"], 6)

    def test_leakage_is_removed(self):
        code, out = self._run(overlap_user=True, quotas=(6, 6, 6))
        self.assertEqual(code, 0)
        report = json.loads((out / "split_report.json").read_text())
        self.assertGreater(report["leakage"]["user_overlap_before"]["train&val"], 0)
        self.assertGreater(report["leakage"]["dropped_for_leakage"], 0)

    def test_split_users_disjoint_after_build(self):
        samples = [ds.Sample("train", "one", "a", "u1", ["0 0.5 0.5 0.1 0.1"]),
                   ds.Sample("val", "one", "b", "u1", ["0 0.5 0.5 0.1 0.1"]),
                   ds.Sample("test", "one", "c", "u2", ["0 0.5 0.5 0.1 0.1"]),
                   ds.Sample("test", "one", "a", "u3", ["0 0.5 0.5 0.1 0.1"])]
        kept, stats = ds.enforce_no_leakage(samples)
        self.assertEqual([(s.split, s.stem) for s in kept], [("train", "a"), ("test", "c")])
        self.assertEqual(stats["dropped_for_leakage"], 2)


class TestRangeReading(unittest.TestCase):
    def test_read_member_matches_zipfile(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.zip"
            with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("x/y.bin", bytes(range(256)) * 400)
            src = ds.LocalRangeSource(p)
            infos = ds.open_zip_index(src)
            self.assertEqual(ds.read_member(src, infos[0]), bytes(range(256)) * 400)
            src.close()


if __name__ == "__main__":
    unittest.main()
