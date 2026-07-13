"""Normalize source-specific dwelling scenarios into app-ready state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class PreparedDwelling:
    """All scenario-owned values that the interactive app installs together."""

    rooms: list
    columns: list
    shafts: list
    covers: list
    doors: list
    walls: list
    wall_polygons: list
    routing_region_base: Any
    shaft_extraction: Any
    terminals: dict
    wet_room_names: list[str]
    wet_room_outer_accents: list
    machine_position: tuple[float, float]
    shaft_core_entry_specs: list[dict[str, Any]]
    label: str
    summary: Mapping[str, Any]


def wet_room_outer_accents(rooms: Iterable, wet_room_names: Iterable[str], *, buffer_mm: float = 10.0) -> list:
    """Create the small outer accents used to identify wet rooms in the UI."""
    wet_names = set(wet_room_names)
    accents = []
    for room in rooms:
        if getattr(room, "name", None) not in wet_names:
            continue
        polygon = getattr(room, "polygon", None)
        if polygon is None or polygon.is_empty:
            continue
        accent = polygon.buffer(buffer_mm, join_style=2)
        if not accent.is_empty:
            accents.append(accent)
    return accents


def prepare_synthetic_dwelling(scenario, *, label: str = "synthetic", summary: Mapping[str, Any] | None = None) -> PreparedDwelling:
    """Adapt the normalized synthetic scenario without app-global mutations."""
    wet_names = list(scenario.wet_room_names)
    return PreparedDwelling(
        rooms=list(scenario.rooms),
        columns=list(scenario.columns),
        shafts=list(scenario.shafts),
        covers=list(scenario.covers),
        doors=list(scenario.doors),
        walls=list(scenario.walls),
        wall_polygons=list(scenario.wall_polygons),
        routing_region_base=scenario.routing_region_base,
        shaft_extraction=scenario.shaft_extraction,
        terminals=dict(scenario.terminals),
        wet_room_names=wet_names,
        wet_room_outer_accents=wet_room_outer_accents(scenario.rooms, wet_names),
        machine_position=tuple(scenario.machine_position),
        shaft_core_entry_specs=[],
        label=label,
        summary=dict(summary or {}),
    )


def prepare_real_dwelling(
    scenario,
    *,
    wall_thickness_mm: float,
    label: str,
    summary: Mapping[str, Any],
    derive_walls: Callable[[list, list, list], list],
    build_wall_polygons: Callable[[list, list, list, float], list],
    choose_machine_position: Callable[[dict, Any], tuple[float, float]],
    build_core_entry_specs: Callable[[Any], list[dict[str, Any]]],
) -> PreparedDwelling:
    """Prepare a real exported dwelling with its app-specific adapters injected."""
    rooms = list(scenario.rooms)
    columns = list(scenario.columns)
    shafts = list(scenario.shafts)
    covers = list(getattr(scenario, "covers", []) or [
        room.polygon for room in rooms if getattr(room, "has_cover", False)
    ])
    terminals = dict(scenario.terminals)
    wet_names = list(terminals)
    walls = derive_walls(rooms, columns, shafts)
    wall_polygons = build_wall_polygons(walls, columns, shafts, wall_thickness_mm)
    shaft_extraction = scenario.shaft_extraction
    return PreparedDwelling(
        rooms=rooms,
        columns=columns,
        shafts=shafts,
        covers=covers,
        doors=[],
        walls=walls,
        wall_polygons=wall_polygons,
        routing_region_base=scenario.routing_region_base,
        shaft_extraction=shaft_extraction,
        terminals=terminals,
        wet_room_names=wet_names,
        wet_room_outer_accents=wet_room_outer_accents(rooms, wet_names),
        machine_position=tuple(choose_machine_position(terminals, shaft_extraction)),
        shaft_core_entry_specs=list(build_core_entry_specs(scenario)),
        label=label,
        summary=dict(summary),
    )
