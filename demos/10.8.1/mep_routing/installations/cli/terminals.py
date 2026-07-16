"""Initial Cli supply-terminal preparation for dwelling rooms."""

from __future__ import annotations

import unicodedata

from shapely.geometry import Point
from shapely.ops import nearest_points


SERVICE_WORDS = ("kitchen", "cocina", "bath", "bano", "toilet", "aseo", "washroom", "lavadero")
TARGET_WORDS = ("bedroom", "dormitorio", "habitacion", "living", "salon", "comedor", "office", "estudio")


def _normalized_room_text(room):
    raw = " ".join(str(getattr(room, name, "") or "") for name in ("name", "category", "type_name"))
    return unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode().lower()


def is_cli_supply_room(room):
    explicit = getattr(room, "clima_is_target", None)
    if explicit is not None:
        return bool(explicit)
    polygon = getattr(room, "polygon", None)
    if polygon is None or polygon.is_empty or not getattr(room, "has_cover", False):
        return False
    text = _normalized_room_text(room)
    return not any(word in text for word in SERVICE_WORDS) and any(word in text for word in TARGET_WORDS)


def prepare_cli_supply_terminals(rooms, routing_region, *, maximum_connection_reach_mm=2000.0):
    """Create one provisional supply-tree terminal per eligible occupied room."""
    eligible = [
        room for room in rooms
        if is_cli_supply_room(room)
        and (routing_region is None or room.polygon.distance(routing_region) <= maximum_connection_reach_mm)
    ]
    if not eligible:
        eligible = [
            room for room in rooms
            if getattr(room, "has_cover", False)
            and getattr(room, "polygon", None) is not None
            and not room.polygon.is_empty
            and not any(word in _normalized_room_text(room) for word in SERVICE_WORDS)
            and (routing_region is None or room.polygon.distance(routing_region) <= maximum_connection_reach_mm)
        ]
    terminals = {}
    counts = {}
    for room in eligible:
        name = str(getattr(room, "name", "") or "Cli").strip() or "Cli"
        counts[name] = counts.get(name, 0) + 1
        unique_name = name if counts[name] == 1 else f"{name} {counts[name]}"
        allowed = room.polygon if routing_region is None else room.polygon.intersection(routing_region)
        if not allowed.is_empty:
            point = allowed.representative_point()
        elif routing_region is not None:
            point = nearest_points(room.polygon, routing_region)[1]
            interior = routing_region.representative_point()
            dx, dy = interior.x - point.x, interior.y - point.y
            length = max((dx * dx + dy * dy) ** 0.5, 1.0)
            candidate = Point(point.x + dx / length, point.y + dy / length)
            point = candidate if routing_region.covers(candidate) else point
        else:
            point = room.polygon.representative_point()
        terminals[f"{unique_name} Supply"] = (round(float(point.x)), round(float(point.y)))
    return terminals
