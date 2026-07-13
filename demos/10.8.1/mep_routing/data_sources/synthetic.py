from __future__ import annotations

from dataclasses import dataclass

from shapely.affinity import scale as shapely_scale
from shapely.geometry import LineString
from shapely.ops import unary_union

from mep_routing.geometry import snap_to_integer_grid

from .dwelling import build_wall_polygons, cut_line_obstacles


DEFAULT_COVERED_ROOM_NAMES = (
    "Hallway",
    "Kitchen",
    "Bathroom",
    "Bathroom 1",
    "Bathroom 2",
    "Toilet",
    "Washroom",
    "Bedroom 1",
)
DEFAULT_WET_ROOM_NAMES = ("Kitchen", "Bathroom", "Toilet", "Washroom")
DEFAULT_MACHINE_ROOM_NAMES = ("Bathroom", "Washroom")


@dataclass
class SyntheticDwelling:
    rooms: list
    columns: list
    shafts: list
    covers: list
    doors: list
    walls: list
    wall_polygons: list
    routing_region_base: object
    shaft_extraction: object | None
    terminals: dict
    wet_room_names: list[str]
    machine_position: tuple[float, float]


def _scaled_geometry(geometry, scale_to_mm):
    return snap_to_integer_grid(
        shapely_scale(geometry, xfact=scale_to_mm, yfact=scale_to_mm, origin=(0, 0))
    )


def _scaled_door(door, scale_to_mm, is_entrance=None):
    return {
        "d1": (round(door["d1"][0] * scale_to_mm), round(door["d1"][1] * scale_to_mm)),
        "d2": (round(door["d2"][0] * scale_to_mm), round(door["d2"][1] * scale_to_mm)),
        "swing_dir": door["swing_dir"],
        "width": door["width"] * scale_to_mm,
        "is_entrance": door.get("is_entrance", False) if is_entrance is None else is_entrance,
    }


def build_synthetic_dwelling(
    layout_provider,
    room_factory,
    representative_point,
    *,
    scale_to_mm=1000,
    wall_thickness_mm=150,
    width_m=15.0,
    height_m=11.0,
    num_rooms=8,
    covered_room_names=DEFAULT_COVERED_ROOM_NAMES,
    wet_room_names=DEFAULT_WET_ROOM_NAMES,
    machine_room_names=DEFAULT_MACHINE_ROOM_NAMES,
    fallback_machine_position=(7500.0, 5500.0),
):
    """Generate and normalize a synthetic dwelling into the shared millimetre model."""
    source_rooms = layout_provider.generate_layout(width=width_m, height=height_m, num_rooms=num_rooms)
    rooms = []
    for source_room in source_rooms:
        room = room_factory(_scaled_geometry(source_room.polygon, scale_to_mm), source_room.name)
        room.has_cover = any(name in source_room.name for name in covered_room_names)
        rooms.append(room)
    covers = [room.polygon for room in rooms if getattr(room, "has_cover", False)]

    source_shafts = layout_provider.generate_mep_shafts(source_rooms)
    shafts = [_scaled_geometry(shaft, scale_to_mm) for shaft in source_shafts]

    source_columns = layout_provider.generate_structural_grid(
        unary_union([room.polygon for room in source_rooms]),
        spacing=4.0,
    )
    columns = []
    for column in source_columns:
        scaled_column = _scaled_geometry(column, scale_to_mm)
        if not any(scaled_column.intersects(shaft) for shaft in shafts):
            columns.append(scaled_column)

    doors = [_scaled_door(door, scale_to_mm) for door in layout_provider.find_door_openings(source_rooms)]
    entrance = layout_provider.find_entrance_door(source_rooms, unary_union([room.polygon for room in source_rooms]))
    if entrance:
        doors.append(_scaled_door(entrance, scale_to_mm, is_entrance=True))

    source_walls = []
    for index, room in enumerate(source_rooms):
        for other_room in source_rooms[index + 1:]:
            shared = room.polygon.intersection(other_room.polygon)
            if isinstance(shared, LineString):
                source_walls.append(shared)
            elif hasattr(shared, "geoms"):
                source_walls.extend(part for part in shared.geoms if isinstance(part, LineString))
    walls = []
    for source_wall in source_walls:
        walls.extend(cut_line_obstacles(_scaled_geometry(source_wall, scale_to_mm), columns, shafts))

    wall_polygons = build_wall_polygons(walls, columns, shafts, wall_thickness_mm)
    routing_region = unary_union([
        room.polygon
        for room in source_rooms
        if any(name in room.name for name in covered_room_names)
    ])
    routing_region_base = _scaled_geometry(routing_region, scale_to_mm)
    for column in columns:
        routing_region_base = routing_region_base.difference(column)
    for shaft in shafts:
        routing_region_base = routing_region_base.difference(shaft)

    shaft_extraction = shafts[0] if shafts else None
    wet_rooms = [room for room in rooms if any(name in room.name for name in wet_room_names)]
    terminals = {room.name: representative_point(room.polygon) for room in wet_rooms}

    machine_room = None
    if shaft_extraction is not None:
        shaft_point = shaft_extraction.representative_point()
        machine_candidates = [
            room
            for room in wet_rooms
            if any(name in room.name for name in machine_room_names)
        ]
        if machine_candidates:
            machine_room = min(
                machine_candidates,
                key=lambda room: abs(shaft_point.x - room.polygon.centroid.x) + abs(shaft_point.y - room.polygon.centroid.y),
            )
    machine_position = representative_point(machine_room.polygon) if machine_room else fallback_machine_position

    return SyntheticDwelling(
        rooms=rooms,
        columns=columns,
        shafts=shafts,
        covers=covers,
        doors=doors,
        walls=walls,
        wall_polygons=wall_polygons,
        routing_region_base=routing_region_base,
        shaft_extraction=shaft_extraction,
        terminals=terminals,
        wet_room_names=list(terminals),
        machine_position=machine_position,
    )
