"""Typed, validated configuration loaded from config/gestures.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "gestures.yaml"

# Gestures may only ever press these keys. There is deliberately no typing or shell access.
ALLOWED_KEYS: frozenset[str] = frozenset({"left", "right", "up", "down", "pageup", "pagedown"})
CAMERA_BACKENDS: frozenset[str] = frozenset({"auto", "dshow", "msmf", "v4l2", "any"})


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    classes: tuple[str, ...]
    confidence: float
    per_class_conf: Mapping[str, float]
    iou: float
    imgsz: int
    device: str

    def threshold_for(self, label: str) -> float:
        return float(self.per_class_conf.get(label, self.confidence))

    @property
    def min_threshold(self) -> float:
        return min([self.confidence, *self.per_class_conf.values()])


@dataclass(frozen=True)
class CameraConfig:
    index: int
    width: int
    height: int
    fps: int
    mirror: bool
    backend: str
    threaded: bool


@dataclass(frozen=True)
class RolesConfig:
    pointer: str
    click: str
    right_click: Optional[str]
    swipe: Optional[str]
    toggle: Optional[str]
    neutral: Optional[str]


@dataclass(frozen=True)
class SmoothingConfig:
    window: int
    min_votes: int
    track_max_dist: float


@dataclass(frozen=True)
class CursorConfig:
    active_region: tuple[float, float, float, float]
    sensitivity: float
    anchor_y: float
    min_cutoff: float
    beta: float
    d_cutoff: float
    deadzone_px: float
    edge_margin_px: int


@dataclass(frozen=True)
class ActionsConfig:
    click_hold_s: float
    right_click_hold_s: float
    click_cooldown_s: float
    click_requires_pointer_context: bool
    pointer_context_s: float
    toggle_hold_s: float
    toggle_cooldown_s: float


@dataclass(frozen=True)
class SwipeConfig:
    enabled: bool
    window_s: float
    min_dx: float
    max_dy_ratio: float
    min_points: int
    monotonic_ratio: float
    cooldown_s: float
    left_key: str
    right_key: str


@dataclass(frozen=True)
class SafetyConfig:
    arm_grace_s: float
    allow_gesture_arming: bool
    max_discrete_actions_per_s: float
    idle_disarm_s: float
    global_hotkeys: bool
    pyautogui_failsafe: bool


@dataclass(frozen=True)
class UIConfig:
    window_name: str
    show_help: bool
    always_on_top: bool


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    camera: CameraConfig
    roles: RolesConfig
    smoothing: SmoothingConfig
    cursor: CursorConfig
    actions: ActionsConfig
    swipe: SwipeConfig
    safety: SafetyConfig
    ui: UIConfig
    source_path: Path = field(default=DEFAULT_CONFIG_PATH)


# --------------------------------------------------------------------------- helpers


def resolve_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository root (absolute paths pass through)."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _section(raw: Mapping[str, Any], name: str, required: set[str], optional: set[str] = frozenset()) -> dict:
    data = raw.get(name)
    if not isinstance(data, Mapping):
        raise ConfigError(f"Config section '{name}' is missing or not a mapping.")
    keys = set(data.keys())
    unknown = keys - required - set(optional)
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{name}': {sorted(unknown)}. Check for typos.")
    missing = required - keys
    if missing:
        raise ConfigError(f"Missing key(s) in '{name}': {sorted(missing)}.")
    return dict(data)


def _num(section: str, key: str, value: Any, lo: float | None = None, hi: float | None = None,
         integer: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{section}.{key} must be a number, got {value!r}.")
    if integer and int(value) != value:
        raise ConfigError(f"{section}.{key} must be an integer, got {value!r}.")
    if lo is not None and value < lo:
        raise ConfigError(f"{section}.{key} must be >= {lo}, got {value}.")
    if hi is not None and value > hi:
        raise ConfigError(f"{section}.{key} must be <= {hi}, got {value}.")
    return int(value) if integer else float(value)


def _bool(section: str, key: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{section}.{key} must be true or false, got {value!r}.")
    return value


def _key(section: str, key: str, value: Any) -> str:
    if value not in ALLOWED_KEYS:
        raise ConfigError(f"{section}.{key}={value!r} is not an allowed key. Allowed: {sorted(ALLOWED_KEYS)}.")
    return str(value)


# --------------------------------------------------------------------------- parsing


def parse_config(raw: Mapping[str, Any], source_path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError("Top level of the config must be a mapping.")
    allowed_top = {"model", "camera", "roles", "smoothing", "cursor", "actions", "swipe", "safety", "ui"}
    unknown = set(raw.keys()) - allowed_top
    if unknown:
        raise ConfigError(f"Unknown top-level section(s): {sorted(unknown)}.")

    m = _section(raw, "model", {"path", "classes", "confidence", "iou", "imgsz", "device"}, {"per_class_conf"})
    classes = m["classes"]
    if not isinstance(classes, list) or not classes or not all(isinstance(c, str) and c for c in classes):
        raise ConfigError("model.classes must be a non-empty list of class names.")
    if len(set(classes)) != len(classes):
        raise ConfigError("model.classes contains duplicates.")
    pcc_raw = m.get("per_class_conf") or {}
    if not isinstance(pcc_raw, Mapping):
        raise ConfigError("model.per_class_conf must be a mapping of class -> threshold.")
    per_class_conf: dict[str, float] = {}
    for label, thr in pcc_raw.items():
        if label not in classes:
            raise ConfigError(f"model.per_class_conf has unknown class '{label}'.")
        per_class_conf[label] = float(_num("model.per_class_conf", label, thr, 0.01, 0.99))
    imgsz = int(_num("model", "imgsz", m["imgsz"], 32, 1920, integer=True))
    if imgsz % 32:
        raise ConfigError(f"model.imgsz must be a multiple of 32, got {imgsz}.")
    model = ModelConfig(
        path=resolve_path(str(m["path"])),
        classes=tuple(classes),
        confidence=float(_num("model", "confidence", m["confidence"], 0.01, 0.99)),
        per_class_conf=per_class_conf,
        iou=float(_num("model", "iou", m["iou"], 0.05, 0.95)),
        imgsz=imgsz,
        device=str(m["device"]),
    )

    c = _section(raw, "camera", {"index", "width", "height", "fps", "mirror", "backend", "threaded"})
    backend = str(c["backend"]).lower()
    if backend not in CAMERA_BACKENDS:
        raise ConfigError(f"camera.backend must be one of {sorted(CAMERA_BACKENDS)}, got {backend!r}.")
    camera = CameraConfig(
        index=int(_num("camera", "index", c["index"], 0, 63, integer=True)),
        width=int(_num("camera", "width", c["width"], 160, 3840, integer=True)),
        height=int(_num("camera", "height", c["height"], 120, 2160, integer=True)),
        fps=int(_num("camera", "fps", c["fps"], 5, 120, integer=True)),
        mirror=_bool("camera", "mirror", c["mirror"]),
        backend=backend,
        threaded=_bool("camera", "threaded", c["threaded"]),
    )

    r = _section(raw, "roles", {"pointer", "click"}, {"right_click", "swipe", "toggle", "neutral"})

    def role(name: str, required: bool) -> Optional[str]:
        value = r.get(name)
        if value is None:
            if required:
                raise ConfigError(f"roles.{name} is required.")
            return None
        if value not in classes:
            raise ConfigError(f"roles.{name}={value!r} is not in model.classes {list(classes)}.")
        return str(value)

    roles = RolesConfig(
        pointer=role("pointer", True),  # type: ignore[arg-type]
        click=role("click", True),  # type: ignore[arg-type]
        right_click=role("right_click", False),
        swipe=role("swipe", False),
        toggle=role("toggle", False),
        neutral=role("neutral", False),
    )
    used = [v for v in (roles.pointer, roles.click, roles.right_click, roles.swipe, roles.toggle, roles.neutral) if v]
    if len(used) != len(set(used)):
        raise ConfigError(f"Each class can have only one role; got duplicates in roles: {used}.")

    s = _section(raw, "smoothing", {"window", "min_votes", "track_max_dist"})
    window = int(_num("smoothing", "window", s["window"], 1, 30, integer=True))
    min_votes = int(_num("smoothing", "min_votes", s["min_votes"], 1, 30, integer=True))
    if min_votes > window or min_votes * 2 <= window:
        raise ConfigError(
            f"smoothing.min_votes ({min_votes}) must be a strict majority of smoothing.window ({window}), "
            f"i.e. between {window // 2 + 1} and {window}."
        )
    smoothing = SmoothingConfig(window, min_votes,
                                float(_num("smoothing", "track_max_dist", s["track_max_dist"], 0.01, 1.5)))

    cu = _section(raw, "cursor", {"active_region", "sensitivity", "anchor_y", "min_cutoff", "beta", "d_cutoff",
                                  "deadzone_px", "edge_margin_px"})
    region = cu["active_region"]
    if (not isinstance(region, list) or len(region) != 4
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in region)):
        raise ConfigError("cursor.active_region must be [x1, y1, x2, y2] in 0..1.")
    x1, y1, x2, y2 = (float(v) for v in region)
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ConfigError(f"cursor.active_region must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1, got {region}.")
    cursor = CursorConfig(
        active_region=(x1, y1, x2, y2),
        sensitivity=float(_num("cursor", "sensitivity", cu["sensitivity"], 0.25, 4.0)),
        anchor_y=float(_num("cursor", "anchor_y", cu["anchor_y"], 0.0, 1.0)),
        min_cutoff=float(_num("cursor", "min_cutoff", cu["min_cutoff"], 0.01, 30.0)),
        beta=float(_num("cursor", "beta", cu["beta"], 0.0, 10.0)),
        d_cutoff=float(_num("cursor", "d_cutoff", cu["d_cutoff"], 0.01, 30.0)),
        deadzone_px=float(_num("cursor", "deadzone_px", cu["deadzone_px"], 0.0, 100.0)),
        edge_margin_px=int(_num("cursor", "edge_margin_px", cu["edge_margin_px"], 2, 200, integer=True)),
    )

    a = _section(raw, "actions", {"click_hold_s", "right_click_hold_s", "click_cooldown_s",
                                  "click_requires_pointer_context", "pointer_context_s", "toggle_hold_s",
                                  "toggle_cooldown_s"})
    actions = ActionsConfig(
        click_hold_s=float(_num("actions", "click_hold_s", a["click_hold_s"], 0.05, 5.0)),
        right_click_hold_s=float(_num("actions", "right_click_hold_s", a["right_click_hold_s"], 0.05, 5.0)),
        click_cooldown_s=float(_num("actions", "click_cooldown_s", a["click_cooldown_s"], 0.1, 10.0)),
        click_requires_pointer_context=_bool("actions", "click_requires_pointer_context",
                                             a["click_requires_pointer_context"]),
        pointer_context_s=float(_num("actions", "pointer_context_s", a["pointer_context_s"], 0.1, 10.0)),
        toggle_hold_s=float(_num("actions", "toggle_hold_s", a["toggle_hold_s"], 0.3, 10.0)),
        toggle_cooldown_s=float(_num("actions", "toggle_cooldown_s", a["toggle_cooldown_s"], 0.3, 30.0)),
    )

    sw = _section(raw, "swipe", {"enabled", "window_s", "min_dx", "max_dy_ratio", "min_points", "monotonic_ratio",
                                 "cooldown_s", "left_key", "right_key"})
    swipe = SwipeConfig(
        enabled=_bool("swipe", "enabled", sw["enabled"]),
        window_s=float(_num("swipe", "window_s", sw["window_s"], 0.1, 3.0)),
        min_dx=float(_num("swipe", "min_dx", sw["min_dx"], 0.05, 1.0)),
        max_dy_ratio=float(_num("swipe", "max_dy_ratio", sw["max_dy_ratio"], 0.05, 2.0)),
        min_points=int(_num("swipe", "min_points", sw["min_points"], 2, 60, integer=True)),
        monotonic_ratio=float(_num("swipe", "monotonic_ratio", sw["monotonic_ratio"], 0.5, 1.0)),
        cooldown_s=float(_num("swipe", "cooldown_s", sw["cooldown_s"], 0.1, 10.0)),
        left_key=_key("swipe", "left_key", sw["left_key"]),
        right_key=_key("swipe", "right_key", sw["right_key"]),
    )

    sa = _section(raw, "safety", {"arm_grace_s", "allow_gesture_arming", "max_discrete_actions_per_s",
                                  "idle_disarm_s", "global_hotkeys", "pyautogui_failsafe"})
    safety = SafetyConfig(
        arm_grace_s=float(_num("safety", "arm_grace_s", sa["arm_grace_s"], 0.0, 10.0)),
        allow_gesture_arming=_bool("safety", "allow_gesture_arming", sa["allow_gesture_arming"]),
        max_discrete_actions_per_s=float(_num("safety", "max_discrete_actions_per_s",
                                              sa["max_discrete_actions_per_s"], 0.2, 20.0)),
        idle_disarm_s=float(_num("safety", "idle_disarm_s", sa["idle_disarm_s"], 0.0, 3600.0)),
        global_hotkeys=_bool("safety", "global_hotkeys", sa["global_hotkeys"]),
        pyautogui_failsafe=_bool("safety", "pyautogui_failsafe", sa["pyautogui_failsafe"]),
    )
    u = _section(raw, "ui", {"window_name", "show_help", "always_on_top"})
    ui = UIConfig(str(u["window_name"]), _bool("ui", "show_help", u["show_help"]),
                  _bool("ui", "always_on_top", u["always_on_top"]))

    return AppConfig(model, camera, roles, smoothing, cursor, actions, swipe, safety, ui, source_path)


def load_config(path: str | Path | None = None) -> AppConfig:
    cfg_path = resolve_path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}. Pass --config or restore config/gestures.yaml.")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {cfg_path}: {exc}") from exc
    return parse_config(raw, cfg_path)


def apply_overrides(cfg: AppConfig, *, camera_index: int | None = None, model_path: str | None = None,
                    confidence: float | None = None, sensitivity: float | None = None) -> AppConfig:
    """Return a copy of the config with CLI overrides applied and re-validated."""
    model, camera, cursor = cfg.model, cfg.camera, cfg.cursor
    if camera_index is not None:
        camera = replace(camera, index=int(_num("--camera", "index", camera_index, 0, 63, integer=True)))
    if model_path is not None:
        model = replace(model, path=resolve_path(model_path))
    if confidence is not None:
        model = replace(model, confidence=float(_num("--conf", "confidence", confidence, 0.01, 0.99)))
    if sensitivity is not None:
        cursor = replace(cursor, sensitivity=float(_num("--sensitivity", "sensitivity", sensitivity, 0.25, 4.0)))
    return replace(cfg, model=model, camera=camera, cursor=cursor)
