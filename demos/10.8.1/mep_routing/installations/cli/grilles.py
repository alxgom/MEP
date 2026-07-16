"""Cli supply/return grille preparation from room and routing geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math

from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from .terminals import is_cli_supply_room


@dataclass(frozen=True)
class CliGrilleConnection:
    route_name: str
    room_name: str
    role: str
    grille_point: tuple[int, int]
    routing_point: tuple[int, int]
    width_mm: int
    inward_normal: tuple[float, float]


@dataclass(frozen=True)
class CliGrillePlan:
    connections: tuple[CliGrilleConnection, ...]

    @property
    def supply(self):
        return tuple(item for item in self.connections if item.role == "supply")

    @property
    def returns(self):
        return tuple(item for item in self.connections if item.role == "return")

    def routing_terminals(self, role):
        return {item.route_name: item.routing_point for item in self.connections if item.role == role}

    def grille_terminals(self):
        return {item.route_name: item.grille_point for item in self.connections}


def _segments(polygon):
    coords = list(polygon.exterior.coords)
    return [
        (tuple(map(float, first)), tuple(map(float, second)), LineString([first, second]))
        for first, second in zip(coords, coords[1:])
        if first != second
    ]


def _inward_normal(room_polygon, first, second):
    dx, dy = second[0] - first[0], second[1] - first[1]
    length = max(math.hypot(dx, dy), 1.0)
    normals = ((-dy / length, dx / length), (dy / length, -dx / length))
    middle = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
    return next((normal for normal in normals if room_polygon.contains(Point(
        middle[0] + normal[0] * 50.0, middle[1] + normal[1] * 50.0,
    ))), normals[0])


def _point_along(first, second, distance):
    dx, dy = second[0] - first[0], second[1] - first[1]
    length = max(math.hypot(dx, dy), 1.0)
    return (round(first[0] + dx * distance / length), round(first[1] + dy * distance / length))


def _routing_point(grille_point, routing_region, inset_mm=300.0):
    if routing_region is None:
        return grille_point
    safe_region = routing_region.buffer(-float(inset_mm), cap_style=3, join_style=2)
    target = safe_region if not safe_region.is_empty else routing_region
    point = nearest_points(Point(grille_point), target)[1]
    return round(float(point.x)), round(float(point.y))


def prepare_cli_grilles(
    rooms, routing_region, *, supply_width_mm=600, return_width_mm=500,
    edge_clearance_mm=150, between_clearance_mm=200, maximum_reach_mm=2000,
):
    """Place one supply and one return grille on an accessible wall per target room."""
    targets = [
        room for room in rooms
        if is_cli_supply_room(room)
        and (routing_region is None or room.polygon.distance(routing_region) <= maximum_reach_mm)
    ]
    connections = []
    counts = {}
    required = supply_width_mm + return_width_mm + 2 * edge_clearance_mm + between_clearance_mm
    for room in targets:
        name = str(getattr(room, "name", "") or "Cli").strip() or "Cli"
        counts[name] = counts.get(name, 0) + 1
        room_name = name if counts[name] == 1 else f"{name} {counts[name]}"
        segments = sorted(
            _segments(room.polygon),
            key=lambda item: (
                item[2].distance(routing_region) if routing_region is not None else 0.0,
                -item[2].length,
            ),
        )
        suitable = next((item for item in segments if item[2].length >= required), None)
        first, second, line = suitable or max(segments, key=lambda item: item[2].length)
        normal = _inward_normal(room.polygon, first, second)
        usable = max(line.length - 2 * edge_clearance_mm, 1.0)
        supply_distance = edge_clearance_mm + min(supply_width_mm / 2.0, usable * 0.25)
        return_distance = line.length - edge_clearance_mm - min(return_width_mm / 2.0, usable * 0.25)
        if return_distance <= supply_distance:
            supply_distance, return_distance = line.length * 0.3, line.length * 0.7
        for role, width, distance in (
            ("supply", supply_width_mm, supply_distance),
            ("return", return_width_mm, return_distance),
        ):
            grille = _point_along(first, second, distance)
            route = f"{room_name} {'Supply' if role == 'supply' else 'Return'}"
            connections.append(CliGrilleConnection(
                route, room_name, role, grille, _routing_point(grille, routing_region),
                width, normal,
            ))
    return CliGrillePlan(tuple(connections))
