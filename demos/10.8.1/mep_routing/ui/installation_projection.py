"""Installation-neutral projection of non-active runs onto the canvas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .frame import CanvasContextLayer


@dataclass(frozen=True)
class InstallationVisual:
    terminals: Mapping[str, tuple[float, float]]
    machine_pins: Mapping[str, tuple[float, float]]


def installation_context_layers(order, enabled, active, runs, visuals):
    return tuple(
        CanvasContextLayer(key, visuals[key].terminals, runs[key].routes or (), visuals[key].machine_pins)
        for key in order
        if key != active and key in enabled and key in runs and key in visuals
    )
