"""Shared test fixtures (plain functions so tests run under both pytest and unittest)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.config import AppConfig, load_config
from src.detector import Detection


def default_config(**section_overrides: dict[str, Any]) -> AppConfig:
    """Load config/gestures.yaml and replace fields per section, e.g. safety={"arm_grace_s": 0}."""
    cfg = load_config()
    for section, fields in section_overrides.items():
        cfg = replace(cfg, **{section: replace(getattr(cfg, section), **fields)})
    return cfg


def det(label: str, conf: float = 0.9, cx: float = 0.5, cy: float = 0.5, size: float = 0.2) -> Detection:
    half = size / 2
    return Detection(label, conf, cx - half, cy - half, cx + half, cy + half)


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t
