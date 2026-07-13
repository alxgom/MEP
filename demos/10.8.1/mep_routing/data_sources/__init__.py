"""Adapters and normalization helpers for dwelling input sources."""

from .dwelling import (
    build_wall_polygons,
    choose_initial_machine_position,
    derive_room_boundary_walls,
)
from .synthetic import SyntheticDwelling, build_synthetic_dwelling

__all__ = [
    "build_wall_polygons",
    "choose_initial_machine_position",
    "derive_room_boundary_walls",
    "SyntheticDwelling",
    "build_synthetic_dwelling",
]
