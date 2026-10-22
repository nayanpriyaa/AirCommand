"""AirCommand entry point.

Run from the repository root:
    python -m src.main                 # live control (starts INACTIVE; press F8 or 'a' to arm)
    python -m src.main --dry-run       # full pipeline, actions only logged, nothing touches the desktop
    python -m src.main --benchmark 300 --dry-run   # measure FPS / latency, write outputs/benchmark_*.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Deque, Optional, Protocol, Sequence

import numpy as np

from src.config import PROJECT_ROOT, AppConfig, ConfigError, apply_overrides, load_config
from src.cursor_controller import ActionExecutor, CursorMapper, DesktopBackend, DesktopError, NullBackend
from src.detector import Detection
from src.gesture_logic import Action, ActionType, GestureEngine, GestureStabilizer, HandSelector, StableGesture
from src.safety import SafetyController

OUTPUTS = PROJECT_ROOT / "outputs"


class DetectorLike(Protocol):
    class_names: Sequence[str]

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


@dataclass
class FrameResult:
    detections: list[Detection]
    primary: Optional[Detection]
    stable: StableGesture
    actions: list[Action]
    events: list[str]
    infer_ms: float


@dataclass
class Stats:
    frame_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=10_000))
    infer_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=10_000))
    frames: int = 0
    started: float = field(default_factory=time.perf_counter)

    def summary(self) -> dict[str, Any]:
        def pct(values: Sequence[float], q: float) -> Optional[float]:
            if not values:
                return None
            s = sorted(values)
            return round(s[min(len(s) - 1, int(q * (len(s) - 1)))], 2)

        fm, im = list(self.frame_ms), list(self.infer_ms)
        return {
            "frames": self.frames,
            "wall_time_s": round(time.perf_counter() - self.started, 2),
            "fps_mean": round(1000.0 / statistics.mean(fm), 2) if fm else None,
            "frame_ms_p50": pct(fm, 0.50),
            "frame_ms_p95": pct(fm, 0.95),
            "inference_ms_mean": round(statistics.mean(im), 2) if im else None,
            "inference_ms_p50": pct(im, 0.50),
            "inference_ms_p95": pct(im, 0.95),
        }


class AirCommandApp:
    """Per-frame pipeline: detect -> select hand -> smooth -> gesture engine -> safety-gated execution.

    Has no camera or GUI dependency, so it is exercised directly by tests with a fake detector.
    """

    def __init__(self, cfg: AppConfig, detector: DetectorLike, backend: DesktopBackend,
                 safety: Optional[SafetyController] = None) -> None:
        self.cfg = cfg
        self.detector = detector
        self.safety = safety or SafetyController(cfg.safety)
        self.selector = HandSelector(cfg.model, cfg.smoothing, cfg.roles.neutral)
        self.stabilizer = GestureStabilizer(cfg.smoothing.window, cfg.smoothing.min_votes)
        self.engine = GestureEngine.from_config(cfg)
        sw, sh = backend.screen_size()
        self.mapper = CursorMapper(cfg.cursor, sw, sh)
        self.executor = ActionExecutor(backend, self.mapper, self.safety)
        self._generation = self.safety.generation
        self.last_event = ""
        self.last_event_t = -1e9

    def process(self, frame: np.ndarray, now: float) -> FrameResult:
        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return self.process_detections(detections, now, infer_ms)

    def process_detections(self, detections: list[Detection], now: float, infer_ms: float = 0.0) -> FrameResult:
        if self.safety.generation != self._generation:
            # Any arm/disarm/e-stop transition: held gestures must be released before acting again.
            self.engine.reset()
            self.mapper.reset()
            self._generation = self.safety.generation
        primary = self.selector.select(detections)
        stable = self.stabilizer.update(primary)
        actions = self.engine.update(primary, stable, now)
        events: list[str] = []
        for action in actions:
            if action.type == ActionType.TOGGLE_ARM:
                self.safety.gesture_toggle(now)
                continue
            desc = self.executor.execute(action, now)
            if desc:
                events.append(desc)
                self.last_event, self.last_event_t = desc, now
        self.safety.tick(primary is not None, now)
        return FrameResult(detections, primary, stable, actions, events, infer_ms)


# --------------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.main", description="AirCommand gesture desktop control")
    p.add_argument("--config", default=None, help="Path to YAML config (default: config/gestures.yaml)")
    p.add_argument("--camera", type=int, default=None, help="Camera index override")
    p.add_argument("--model", default=None, help="Weights path override (default from config: models/best.pt)")
    p.add_argument("--conf", type=float, default=None, help="Global confidence threshold override")
    p.add_argument("--sensitivity", type=float, default=None, help="Cursor sensitivity override (0.25-4)")
    p.add_argument("--source", default=None, help="Video file instead of webcam (for testing/demo)")
    p.add_argument("--dry-run", action="store_true", help="Never touch the desktop; log actions only")
    p.add_argument("--headless", action="store_true", help="No preview window (benchmark/video only)")
    p.add_argument("--benchmark", type=int, default=0, metavar="N", help="Stop after N frames and save stats")
    p.add_argument("--record", default=None, metavar="PATH", help="Save the annotated preview to an .mp4")
    p.add_argument("--no-hotkeys", action="store_true", help="Disable global Esc/F8/F9 hotkeys")
    return p


def _package_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("ultralytics", "torch", "opencv-python", "numpy", "pyautogui", "pynput"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def _write_json(prefix: str, payload: dict[str, Any]) -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        cfg = apply_overrides(load_config(args.config), camera_index=args.camera, model_path=args.model,
                              confidence=args.conf, sensitivity=args.sensitivity)
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    # Heavy / platform-specific imports happen only after the config is valid.
    import cv2

    from src import overlay
    from src.camera import Camera, CameraError
    from src.detector import GestureDetector, ModelLoadError
    from src.safety import GlobalHotkeys

    try:
        detector = GestureDetector(cfg.model)
    except ModelLoadError as exc:
        print(f"[model] {exc}", file=sys.stderr)
        return 3

    backend: DesktopBackend
    if args.dry_run:
        backend = NullBackend(verbose=True)
    else:
        try:
            from src.cursor_controller import PyAutoGuiBackend
            backend = PyAutoGuiBackend(failsafe=cfg.safety.pyautogui_failsafe)
        except DesktopError as exc:
            print(f"[desktop] {exc}", file=sys.stderr)
            return 4

    # For --source replay, every component shares the video timeline (see main loop).
    replay_clock = [time.monotonic()]
    clock = (lambda: replay_clock[0]) if args.source else time.monotonic
    app = AirCommandApp(cfg, detector, backend, SafetyController(cfg.safety, clock=clock))
    hotkeys = GlobalHotkeys(app.safety)
    hotkeys_ok = False
    if cfg.safety.global_hotkeys and not args.no_hotkeys:
        hotkeys_ok = hotkeys.start()
        if not hotkeys_ok:
            print(f"[hotkeys] {hotkeys.error}", file=sys.stderr)

    camera = Camera(cfg.camera, source=args.source)
    try:
        camera.open()
    except CameraError as exc:
        print(f"[camera] {exc}", file=sys.stderr)
        hotkeys.stop()
        return 5

    res = camera.resolution
    print(f"[info] camera: {res[0]}x{res[1]} via {camera.backend_name}; model: {cfg.model.path}")
    print(f"[info] screen: {app.mapper.screen_w}x{app.mapper.screen_h}; dry-run: {args.dry_run}")
    print("[info] Desktop control is INACTIVE. Press F8 or 'a' to arm, Esc for emergency stop, 'q' to quit.")
    detector.warmup((max(res[1], 120), max(res[0], 160)))

    window = cfg.ui.window_name
    show_window = not args.headless
    if show_window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        if cfg.ui.always_on_top and hasattr(cv2, "WND_PROP_TOPMOST"):
            try:
                cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
            except cv2.error:
                pass

    writer = None
    stats = Stats()
    show_help = cfg.ui.show_help
    fps_ema = 0.0
    exit_code = 0
    roles = {"pointer": cfg.roles.pointer, "click": cfg.roles.click, "swipe": cfg.roles.swipe}
    # Video files are processed as fast as possible, so wall-clock time would compress every hold
    # and swipe window. Replay uses the file's own timeline instead.
    video_t0, video_fps = replay_clock[0], camera.file_fps
    try:
        while True:
            t_frame = time.perf_counter()
            frame = camera.read()
            if frame is None:  # end of video file
                break
            if camera.is_file:
                now = replay_clock[0] = video_t0 + stats.frames / video_fps
            else:
                now = time.monotonic()
            result = app.process(frame, now)
            for ev in result.events:
                print(f"[action] {ev}")

            msg, msg_t = app.safety.message
            data = overlay.OverlayData(
                state=app.safety.state, message=msg, message_age=now - msg_t,
                detections=result.detections, primary=result.primary, stable=result.stable,
                status=app.engine.status, roles=roles, class_names=detector.class_names,
                active_region=app.mapper.effective_region, anchor_y=cfg.cursor.anchor_y,
                fps=fps_ema, infer_ms=result.infer_ms, last_event=app.last_event,
                last_event_age=now - app.last_event_t, dry_run=args.dry_run, show_help=show_help,
                hotkeys_ok=hotkeys_ok or args.headless,
            )
            vis = overlay.draw(frame, data)

            if args.record:
                if writer is None:
                    rec = Path(args.record)
                    rec.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(rec), cv2.VideoWriter_fourcc(*"mp4v"), float(cfg.camera.fps),
                                             (vis.shape[1], vis.shape[0]))
                writer.write(vis)

            if show_window:
                cv2.imshow(window, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == 27:
                    app.safety.emergency_stop("Esc (window)")
                elif key == ord("a"):
                    app.safety.arm(source="key a")
                elif key == ord("d"):
                    app.safety.disarm("key d")
                elif key == ord("r"):
                    app.safety.reset_estop()
                elif key == ord("h"):
                    show_help = not show_help
                elif key == ord("s"):
                    OUTPUTS.mkdir(parents=True, exist_ok=True)
                    shot = OUTPUTS / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(str(shot), vis)
                    print(f"[info] saved {shot}")
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break  # window closed with the mouse

            dt_ms = (time.perf_counter() - t_frame) * 1000.0
            stats.frames += 1
            stats.frame_ms.append(dt_ms)
            stats.infer_ms.append(result.infer_ms)
            fps_ema = 1000.0 / dt_ms if fps_ema == 0 else 0.9 * fps_ema + 0.1 * (1000.0 / max(dt_ms, 1e-3))
            if args.benchmark and stats.frames >= args.benchmark:
                break
    except KeyboardInterrupt:
        print("\n[info] interrupted")
    except CameraError as exc:
        print(f"[camera] {exc}", file=sys.stderr)
        exit_code = 5
    finally:
        app.safety.emergency_stop("shutdown")
        hotkeys.stop()
        camera.release()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()

    summary = {
        "mode": "benchmark" if args.benchmark else "session",
        "dry_run": args.dry_run,
        "source": args.source or f"camera:{cfg.camera.index}",
        "resolution": list(res),
        "model": str(cfg.model.path),
        "device": cfg.model.device,
        "imgsz": cfg.model.imgsz,
        **stats.summary(),
        "action_counts": app.executor.counts,
        "versions": _package_versions(),
        "note": "frame_ms includes capture wait, inference, logic, drawing and display.",
    }
    path = _write_json("benchmark" if args.benchmark else "session", summary)
    print(f"[info] stats saved to {path}")
    print(json.dumps({k: summary[k] for k in ("frames", "fps_mean", "inference_ms_p50", "inference_ms_p95")}))
    return exit_code


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
