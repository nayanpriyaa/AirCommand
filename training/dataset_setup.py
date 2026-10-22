"""Build a YOLO-format AirCommand dataset from HaGRIDv2 without downloading 100+ GB.

Source dataset: HaGRIDv2 (https://github.com/hukenovs/hagrid), license: a variant of
CC BY-SA 4.0 (see the repo's license/ folder). Official splits are by subject (user_id).

The 512px HaGRIDv2 archive is a single ~119 GB ZIP. Instead of downloading it, this script
reads the ZIP's central directory with HTTP Range requests and fetches only the sampled
images (~1-2 GB for the defaults). If the server ever stops honouring Range requests, use
--images-zip/--annotations-zip (local archives) or --images-dir/--annotations-dir
(already-extracted folders) instead.

Usage (Colab or Linux):
    python training/dataset_setup.py --out /content/datasets/aircommand
Usage (PowerShell, from repo root):
    python training/dataset_setup.py --out datasets/aircommand
"""

from __future__ import annotations

import argparse
import io
import json
import random
import struct
import sys
import threading
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Optional, Protocol

REPO_ROOT = Path(__file__).resolve().parent.parent

# Links published in the HaGRIDv2 README (https://github.com/hukenovs/hagrid#downloads).
IMAGES_512_URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagridv2_512.zip"
ANNOTATIONS_URL = ("https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/"
                   "annotations_with_landmarks/annotations.zip")

SPLITS = ("train", "val", "test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
DEFAULT_CLASSES = ("one", "fist", "palm", "like", "peace", "no_gesture")


class DatasetError(RuntimeError):
    pass


# --------------------------------------------------------------------------- byte-range sources


class RangeSource(Protocol):
    size: int
    description: str

    def read(self, start: int, length: int) -> bytes: ...


class LocalRangeSource:
    """Range reads from a local file (used for local archives and in tests)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise DatasetError(f"Archive not found: {self.path}")
        self.size = self.path.stat().st_size
        self.description = str(self.path)
        self._local = threading.local()
        self._handles: list[io.BufferedReader] = []
        self._handles_lock = threading.Lock()

    def read(self, start: int, length: int) -> bytes:
        fh = getattr(self._local, "fh", None)
        if fh is None:
            fh = self._local.fh = open(self.path, "rb")
            with self._handles_lock:
                self._handles.append(fh)
        fh.seek(start)
        return fh.read(length)

    def close(self) -> None:
        with self._handles_lock:
            for fh in self._handles:
                fh.close()
            self._handles.clear()
        self._local = threading.local()


def close_sources(*sources: object) -> None:
    for src in sources:
        close = getattr(src, "close", None)
        if callable(close):
            close()


class HttpRangeSource:
    """Range reads over HTTP(S) with per-thread sessions and retries."""

    def __init__(self, url: str, timeout: float = 60.0, retries: int = 5) -> None:
        try:
            import requests
        except ImportError as exc:
            raise DatasetError("pip install requests") from exc
        self._requests = requests
        self.url = url
        self.description = url
        self.timeout = timeout
        self.retries = retries
        self._local = threading.local()
        resp = self._session().get(url, headers={"Range": "bytes=0-0"}, timeout=timeout, stream=True)
        resp.close()
        if resp.status_code != 206 or "Content-Range" not in resp.headers:
            raise DatasetError(
                f"{url} did not honour an HTTP Range request (status {resp.status_code}).\n"
                "Download the archive manually and pass --images-zip / --annotations-zip, "
                "or extract it and pass --images-dir / --annotations-dir."
            )
        self.size = int(resp.headers["Content-Range"].split("/")[-1])

    def _session(self):
        s = getattr(self._local, "session", None)
        if s is None:
            s = self._local.session = self._requests.Session()
        return s

    def read(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b""
        end = min(start + length, self.size) - 1
        last_exc: Optional[BaseException] = None
        for attempt in range(self.retries):
            try:
                resp = self._session().get(self.url, headers={"Range": f"bytes={start}-{end}"},
                                           timeout=self.timeout)
                if resp.status_code == 206:
                    data = resp.content
                    if len(data) == end - start + 1:
                        return data
                    last_exc = DatasetError(f"short read {len(data)} != {end - start + 1}")
                elif resp.status_code == 200:
                    raise DatasetError("Server ignored the Range header mid-run; use a local archive instead.")
                else:
                    last_exc = DatasetError(f"HTTP {resp.status_code}")
            except DatasetError:
                raise
            except Exception as exc:  # connection reset, timeout, ...
                last_exc = exc
            time.sleep(min(30.0, 1.5 * 2 ** attempt))
        raise DatasetError(f"Range read failed after {self.retries} attempts: {last_exc}")


class RangeFile(io.RawIOBase):
    """Seekable read-only file object over a RangeSource, so ``zipfile`` can parse the central directory."""

    def __init__(self, source: RangeSource) -> None:
        super().__init__()
        self._src = source
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._src.size + offset
        self._pos = max(0, self._pos)
        return self._pos

    def readinto(self, b) -> int:  # type: ignore[override]
        n = min(len(b), self._src.size - self._pos)
        if n <= 0:
            return 0
        data = self._src.read(self._pos, n)
        b[: len(data)] = data
        self._pos += len(data)
        return len(data)


_LOCAL_HEADER = struct.Struct("<4s2B4HL2L2H")  # same layout as zipfile.structFileHeader
_LOCAL_SIG = b"PK\x03\x04"


def read_member(source: RangeSource, info: zipfile.ZipInfo) -> bytes:
    """Fetch and decompress one ZIP member with (usually) a single range request. Thread-safe."""
    if info.flag_bits & 0x1:
        raise DatasetError(f"{info.filename} is encrypted")
    guess = _LOCAL_HEADER.size + len(info.filename.encode("utf-8")) + 256 + info.compress_size
    blob = source.read(info.header_offset, guess)
    if len(blob) < _LOCAL_HEADER.size or blob[:4] != _LOCAL_SIG:
        raise DatasetError(f"Bad local header for {info.filename}")
    fields = _LOCAL_HEADER.unpack(blob[: _LOCAL_HEADER.size])
    name_len, extra_len = fields[10], fields[11]
    start = _LOCAL_HEADER.size + name_len + extra_len
    end = start + info.compress_size
    if end > len(blob):
        blob += source.read(info.header_offset + len(blob), end - len(blob))
    payload = blob[start:end]
    if info.compress_type == zipfile.ZIP_STORED:
        data = payload
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(payload, -15)
    else:
        raise DatasetError(f"Unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
        raise DatasetError(f"CRC mismatch for {info.filename} (corrupt transfer)")
    return data


def open_zip_index(source: RangeSource) -> list[zipfile.ZipInfo]:
    print(f"[zip] reading central directory of {source.description} ({source.size / 1e9:.1f} GB)...", flush=True)
    t0 = time.time()
    raw = io.BufferedReader(RangeFile(source), buffer_size=4 << 20)
    with zipfile.ZipFile(raw) as zf:  # type: ignore[arg-type]
        infos = zf.infolist()
    print(f"[zip] {len(infos):,} entries in {time.time() - t0:.0f}s", flush=True)
    return infos


# --------------------------------------------------------------------------- annotation / image access


class Store(Protocol):
    def annotations(self, split: str, cls: str) -> dict[str, Any]: ...
    def has_image(self, cls: str, stem: str) -> bool: ...
    def image_bytes(self, cls: str, stem: str) -> tuple[bytes, str]: ...


@dataclass
class ZipStore:
    images: RangeSource
    annotations_src: RangeSource
    classes: tuple[str, ...]
    _img_index: dict[tuple[str, str], zipfile.ZipInfo] = field(default_factory=dict)
    _ann_index: dict[tuple[str, str], zipfile.ZipInfo] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wanted = set(self.classes)
        for info in open_zip_index(self.annotations_src):
            p = PurePosixPath(info.filename)
            if len(p.parts) >= 2 and p.parent.name in SPLITS and p.suffix == ".json" and p.stem in wanted:
                self._ann_index[(p.parent.name, p.stem)] = info
        for info in open_zip_index(self.images):
            if info.is_dir():
                continue
            p = PurePosixPath(info.filename)
            if p.suffix.lower() in IMAGE_EXTS and p.parent.name in wanted:
                self._img_index[(p.parent.name, p.stem)] = info
        found = Counter(cls for cls, _ in self._img_index)
        print(f"[zip] images found per class: {dict(found)}", flush=True)
        missing = [c for c in self.classes if found[c] == 0]
        if missing:
            raise DatasetError(f"No images for classes {missing} in the image archive. "
                               "Run with --list-only to inspect its layout.")

    def annotations(self, split: str, cls: str) -> dict[str, Any]:
        info = self._ann_index.get((split, cls))
        if info is None:
            raise DatasetError(f"Annotation file {split}/{cls}.json not found in the annotation archive.")
        return json.loads(read_member(self.annotations_src, info).decode("utf-8"))

    def has_image(self, cls: str, stem: str) -> bool:
        return (cls, stem) in self._img_index

    def image_bytes(self, cls: str, stem: str) -> tuple[bytes, str]:
        info = self._img_index[(cls, stem)]
        return read_member(self.images, info), PurePosixPath(info.filename).suffix.lower()


@dataclass
class DirStore:
    images_dir: Path
    annotations_dir: Path

    def annotations(self, split: str, cls: str) -> dict[str, Any]:
        path = self.annotations_dir / split / f"{cls}.json"
        if not path.is_file():
            raise DatasetError(f"Missing annotation file {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _find(self, cls: str, stem: str) -> Optional[Path]:
        for ext in IMAGE_EXTS:
            p = self.images_dir / cls / f"{stem}{ext}"
            if p.is_file():
                return p
        return None

    def has_image(self, cls: str, stem: str) -> bool:
        return self._find(cls, stem) is not None

    def image_bytes(self, cls: str, stem: str) -> tuple[bytes, str]:
        p = self._find(cls, stem)
        if p is None:
            raise DatasetError(f"Image {cls}/{stem} not found under {self.images_dir}")
        return p.read_bytes(), p.suffix.lower()


# --------------------------------------------------------------------------- conversion


def hagrid_to_yolo_lines(record: dict[str, Any], class_to_id: dict[str, int]) -> Optional[list[str]]:
    """Convert one HaGRID record (normalized top-left x, y, w, h) to YOLO lines.

    Returns None if the image contains a hand label outside our vocabulary: keeping such an
    image would teach the model that a real hand is background.
    """
    bboxes, labels = record.get("bboxes") or [], record.get("labels") or []
    if not bboxes or len(bboxes) != len(labels):
        return None
    lines: list[str] = []
    for box, label in zip(bboxes, labels):
        if label not in class_to_id:
            return None
        x, y, w, h = (float(v) for v in box)
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(1.0, x + w), min(1.0, y + h)
        if x2 - x1 <= 1e-4 or y2 - y1 <= 1e-4:
            continue
        lines.append(f"{class_to_id[label]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return lines or None


@dataclass
class Sample:
    split: str
    cls: str
    stem: str
    user_id: str
    lines: list[str]

    @property
    def name(self) -> str:
        return f"{self.cls}__{self.stem}"


def select_samples(store: Store, classes: tuple[str, ...], quotas: dict[str, int], seed: int,
                   log: Callable[[str], None] = print) -> tuple[list[Sample], dict[str, Any]]:
    class_to_id = {c: i for i, c in enumerate(classes)}
    samples: list[Sample] = []
    stats: dict[str, Any] = {"skipped_foreign_label": 0, "skipped_missing_image": 0, "shortfall": {}}
    for split in SPLITS:
        for cls in classes:
            ann = store.annotations(split, cls)
            ids = sorted(ann.keys())
            random.Random(f"{seed}-{split}-{cls}").shuffle(ids)
            chosen = 0
            for stem in ids:
                if chosen >= quotas[split]:
                    break
                lines = hagrid_to_yolo_lines(ann[stem], class_to_id)
                if lines is None:
                    stats["skipped_foreign_label"] += 1
                    continue
                if not store.has_image(cls, stem):
                    stats["skipped_missing_image"] += 1
                    continue
                samples.append(Sample(split, cls, stem, str(ann[stem].get("user_id", "")), lines))
                chosen += 1
            if chosen < quotas[split]:
                stats["shortfall"][f"{split}/{cls}"] = quotas[split] - chosen
            log(f"[select] {split:5s} {cls:12s} {chosen:6d} images (of {len(ids):,} annotated)")
    return samples, stats


def enforce_no_leakage(samples: list[Sample]) -> tuple[list[Sample], dict[str, Any]]:
    """Guarantee no subject (user_id) and no image appears in more than one split.

    HaGRID's official splits are already subject-disjoint; this is a verification step that
    also repairs the dataset (train wins, then val) if that assumption is ever violated.
    """
    users: dict[str, set[str]] = defaultdict(set)
    for s in samples:
        if s.user_id:
            users[s.split].add(s.user_id)
    overlap = {
        "train&val": len(users["train"] & users["val"]),
        "train&test": len(users["train"] & users["test"]),
        "val&test": len(users["val"] & users["test"]),
    }
    banned_val = users["train"]
    banned_test = users["train"] | users["val"]
    seen_stems: set[str] = set()
    kept: list[Sample] = []
    dropped = 0
    for s in sorted(samples, key=lambda s: SPLITS.index(s.split)):
        if s.stem in seen_stems:
            dropped += 1
            continue
        if (s.split == "val" and s.user_id in banned_val) or (s.split == "test" and s.user_id in banned_test):
            dropped += 1
            continue
        seen_stems.add(s.stem)
        kept.append(s)
    return kept, {"user_overlap_before": overlap, "dropped_for_leakage": dropped,
                  "users_per_split": {k: len(v) for k, v in users.items()}}


def write_dataset(store: Store, samples: list[Sample], out: Path, workers: int) -> dict[str, int]:
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    counters = Counter()
    lock = threading.Lock()

    def job(s: Sample) -> str:
        label_path = out / "labels" / s.split / f"{s.name}.txt"
        existing = [p for p in (out / "images" / s.split).glob(f"{s.name}.*")]
        if existing and label_path.is_file():
            return "cached"
        data, ext = store.image_bytes(s.cls, s.stem)
        img_path = out / "images" / s.split / f"{s.name}{ext}"
        tmp = img_path.with_suffix(img_path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(img_path)
        label_path.write_text("\n".join(s.lines) + "\n", encoding="utf-8")
        return "downloaded"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, s): s for s in samples}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                result = fut.result()
            except Exception as exc:
                result = "failed"
                print(f"[write] failed {futures[fut].name}: {exc}", flush=True)
            with lock:
                counters[result] += 1
            if i % 500 == 0 or i == len(samples):
                rate = i / max(1e-6, time.time() - t0)
                print(f"[write] {i:,}/{len(samples):,}  ({rate:.1f} img/s)  {dict(counters)}", flush=True)
    return dict(counters)


def remove_orphans(out: Path, samples: Iterable[Sample]) -> int:
    """Delete images/labels from earlier runs with different quotas so splits stay exact."""
    keep = {(s.split, s.name) for s in samples}
    removed = 0
    for split in SPLITS:
        for sub in ("images", "labels"):
            folder = out / sub / split
            if not folder.is_dir():
                continue
            for p in folder.iterdir():
                stem = p.name.split(".")[0]
                if (split, stem) not in keep:
                    p.unlink()
                    removed += 1
    return removed


def write_data_yaml(out: Path, classes: tuple[str, ...]) -> Path:
    lines = [f"path: {out.resolve().as_posix()}", "train: images/train", "val: images/val", "test: images/test",
             "names:"] + [f"  {i}: {c}" for i, c in enumerate(classes)]
    path = out / "data.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def summarize(out: Path, classes: tuple[str, ...]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split in SPLITS:
        boxes = Counter()
        n_img = 0
        for lbl in (out / "labels" / split).glob("*.txt"):
            n_img += 1
            for line in lbl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    boxes[classes[int(line.split()[0])]] += 1
        n_files = sum(1 for p in (out / "images" / split).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if n_files != n_img:
            raise DatasetError(f"{split}: {n_files} images but {n_img} label files")
        report[split] = {"images": n_img, "boxes": {c: boxes.get(c, 0) for c in classes}}
    return report


# --------------------------------------------------------------------------- CLI


def default_classes() -> tuple[str, ...]:
    cfg = REPO_ROOT / "config" / "gestures.yaml"
    try:
        import yaml

        return tuple(yaml.safe_load(cfg.read_text(encoding="utf-8"))["model"]["classes"])
    except Exception:
        return DEFAULT_CLASSES


def build_store(args: argparse.Namespace, classes: tuple[str, ...]) -> Store:
    if args.images_dir or args.annotations_dir:
        if not (args.images_dir and args.annotations_dir):
            raise DatasetError("--images-dir and --annotations-dir must be used together.")
        return DirStore(Path(args.images_dir), Path(args.annotations_dir))
    images = LocalRangeSource(Path(args.images_zip)) if args.images_zip else HttpRangeSource(args.images_url)
    anns = (LocalRangeSource(Path(args.annotations_zip)) if args.annotations_zip
            else HttpRangeSource(args.annotations_url))
    return ZipStore(images, anns, classes)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(REPO_ROOT / "datasets" / "aircommand"), help="Output dataset folder")
    p.add_argument("--classes", nargs="+", default=None, help="HaGRID class names (default: config model.classes)")
    p.add_argument("--train-per-class", type=int, default=2500)
    p.add_argument("--val-per-class", type=int, default=400)
    p.add_argument("--test-per-class", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16, help="Parallel downloads")
    p.add_argument("--images-url", default=IMAGES_512_URL)
    p.add_argument("--annotations-url", default=ANNOTATIONS_URL)
    p.add_argument("--images-zip", default=None, help="Local copy of hagridv2_512.zip")
    p.add_argument("--annotations-zip", default=None, help="Local copy of annotations.zip")
    p.add_argument("--images-dir", default=None, help="Extracted images root (<class>/<id>.jpg)")
    p.add_argument("--annotations-dir", default=None, help="Extracted annotations root (<split>/<class>.json)")
    p.add_argument("--list-only", action="store_true", help="Print archive layout samples and exit")
    args = p.parse_args(argv)

    classes = tuple(args.classes) if args.classes else default_classes()
    out = Path(args.out)
    quotas = {"train": args.train_per_class, "val": args.val_per_class, "test": args.test_per_class}
    try:
        if args.list_only:
            for url_or_path in ((args.annotations_zip or args.annotations_url), (args.images_zip or args.images_url)):
                src = LocalRangeSource(Path(url_or_path)) if Path(url_or_path).is_file() else HttpRangeSource(url_or_path)
                infos = open_zip_index(src)
                for info in infos[:15]:
                    print("   ", info.filename, info.file_size)
                tops = Counter(PurePosixPath(i.filename).parts[0] for i in infos)
                print("    top-level entries:", tops.most_common(10))
            return 0

        store = build_store(args, classes)
        try:
            return _build(store, args, classes, out, quotas)
        finally:
            close_sources(getattr(store, "images", None), getattr(store, "annotations_src", None))
    except DatasetError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


def _build(store: Store, args: argparse.Namespace, classes: tuple[str, ...], out: Path,
           quotas: dict[str, int]) -> int:
    samples, sel_stats = select_samples(store, classes, quotas, args.seed)
    samples, leak_stats = enforce_no_leakage(samples)
    print(f"[leakage] {leak_stats}", flush=True)
    write_stats = write_dataset(store, samples, out, args.workers)
    if write_stats.get("failed"):
        print(f"[warn] {write_stats['failed']} images failed. Re-run the same command to retry them "
              "(finished files are cached).", flush=True)
    removed = remove_orphans(out, samples)
    data_yaml = write_data_yaml(out, classes)
    report = {
        "source": "HaGRIDv2 (https://github.com/hukenovs/hagrid), CC BY-SA 4.0 variant",
        "classes": list(classes),
        "quotas": quotas,
        "seed": args.seed,
        "selection": sel_stats,
        "leakage": leak_stats,
        "write": write_stats,
        "orphans_removed": removed,
        "splits": summarize(out, classes),
    }
    (out / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["splits"], indent=2))
    print(f"[done] {data_yaml}")
    return 1 if write_stats.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
