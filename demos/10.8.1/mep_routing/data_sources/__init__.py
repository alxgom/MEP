"""Adapters and normalization helpers for dwelling input sources."""

from .dwelling import (
    build_wall_polygons,
    choose_initial_machine_position,
    derive_room_boundary_walls,
)
from .catalog import DwellingCase, discover_dwelling_cases, dwelling_cases_from_rows
from .synthetic import SyntheticDwelling, build_synthetic_dwelling
from .scenario import PreparedDwelling, prepare_real_dwelling, prepare_synthetic_dwelling, wet_room_outer_accents

__all__ = [
    "build_wall_polygons",
    "choose_initial_machine_position",
    "derive_room_boundary_walls",
    "DwellingCase",
    "discover_dwelling_cases",
    "dwelling_cases_from_rows",
    "SyntheticDwelling",
    "build_synthetic_dwelling",
    "PreparedDwelling",
    "prepare_real_dwelling",
    "prepare_synthetic_dwelling",
    "wet_room_outer_accents",
]
