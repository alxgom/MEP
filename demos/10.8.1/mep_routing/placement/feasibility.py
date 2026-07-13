from __future__ import annotations

from shapely.geometry import Point

from .scoring import machine_polygon_from_pins


def is_machine_placement_valid(cx, cy, pins, routing_region, walls, columns, shafts):
    machine_poly = machine_polygon_from_pins(pins)

    if not routing_region or not routing_region.contains(Point(cx, cy)):
        return False
    if any(machine_poly.intersects(w) for w in walls):
        return False
    if any(machine_poly.intersects(col) for col in columns):
        return False
    if any(machine_poly.intersects(s) for s in shafts):
        return False
    return True


def candidate_machine_rooms(rooms, min_area):
    candidates = [
        room
        for room in rooms
        if getattr(room, "has_cover", False)
        and hasattr(room, "polygon")
        and not room.polygon.is_empty
        and room.polygon.area >= min_area
    ]
    return candidates or [
        room
        for room in rooms
        if hasattr(room, "polygon") and not room.polygon.is_empty
    ]
