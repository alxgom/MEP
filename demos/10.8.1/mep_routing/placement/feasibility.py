from __future__ import annotations

from shapely.geometry import Point

from .scoring import machine_polygon_from_pins


def is_machine_placement_valid(
    cx,
    cy,
    pins,
    routing_region,
    walls,
    columns,
    shafts,
    blocked_vertical_regions=(),
    room_regions=(),
):
    machine_poly = machine_polygon_from_pins(pins)

    if (
        routing_region is None
        or routing_region.is_empty
        or not routing_region.covers(Point(cx, cy))
        or not routing_region.covers(machine_poly)
    ):
        return False
    if room_regions and not any(room.covers(machine_poly) for room in room_regions if not room.is_empty):
        return False
    if any(machine_poly.intersects(w) for w in walls):
        return False
    if any(machine_poly.intersects(col) for col in columns):
        return False
    if any(machine_poly.intersects(s) for s in shafts):
        return False
    if any(machine_poly.intersection(region).area > 1e-7 for region in blocked_vertical_regions):
        return False
    return True


def candidate_machine_rooms(rooms, min_area, placeable_region=None):
    def available_area(room):
        polygon = getattr(room, "polygon", None)
        if polygon is None or polygon.is_empty:
            return 0.0
        return polygon.area if placeable_region is None else polygon.intersection(placeable_region).area

    candidates = [
        room
        for room in rooms
        if getattr(room, "has_cover", False)
        and hasattr(room, "polygon")
        and not room.polygon.is_empty
        and available_area(room) >= min_area
    ]
    return candidates or [
        room
        for room in rooms
        if hasattr(room, "polygon")
        and not room.polygon.is_empty
        and available_area(room) > 1e-7
    ]
