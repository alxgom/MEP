"""Cli grille placement ported from the legacy 10.8_cli prototype."""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
import math

from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union

from mep_routing.domain.machines import local_axis_to_world

from .terminals import is_cli_supply_room


@dataclass(frozen=True)
class CliGrillePlacementConfig:
    supply_width_mm: int = 1000
    return_width_mm: int = 1050
    minimum_wall_length_mm: int = 500
    minimum_wall_distance_mm: int = 50
    minimum_between_mm: int = 0
    maximum_overlap_mm: int = 0
    allowed_region_buffer_mm: int = 300
    minimum_false_ceiling_area_mm2: int = 500_000
    connector_length_mm: int = 400


@dataclass(frozen=True)
class CliGrilleConnection:
    route_name: str
    room_name: str
    role: str
    grille_point: tuple[int, int]
    routing_point: tuple[int, int]
    width_mm: int
    inward_normal: tuple[float, float]
    placement_type: str = "wall"
    room_area_m2: float = 0.0
    sizing: object | None = None


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


@dataclass(frozen=True)
class _Candidate:
    point: tuple[int, int]
    routing_point: tuple[int, int]
    role: str
    inward_normal: tuple[float, float]
    segment_index: int
    segment_length: float
    distance_along: float
    width_mm: int
    placement_type: str = "wall"


def _point_allowed(room, point, routing_region, buffer_mm):
    candidate = Point(point)
    if not room.polygon.covers(candidate):
        return False
    if routing_region is None:
        return True
    return routing_region.buffer(buffer_mm, cap_style=2, join_style=3).covers(candidate)


def _routing_allowed(point, routing_region):
    return routing_region is None or routing_region.covers(Point(point))


def _unit_and_inward(first, second, room_polygon):
    dx, dy = second[0] - first[0], second[1] - first[1]
    length = math.hypot(dx, dy)
    if length <= 1e-7:
        return None
    unit = (dx / length, dy / length)
    normals = ((-unit[1], unit[0]), (unit[1], -unit[0]))
    middle = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
    inward = next((normal for normal in normals if room_polygon.contains(Point(
        middle[0] + normal[0] * 80.0, middle[1] + normal[1] * 80.0,
    ))), normals[0])
    return unit, inward, length


def _candidate(room, segment, index, distance, width, role, routing_region, config):
    vector = _unit_and_inward(segment["p0"], segment["p1"], room.polygon)
    if vector is None:
        return None
    unit, inward, length = vector
    point = (
        round(segment["p0"][0] + unit[0] * distance),
        round(segment["p0"][1] + unit[1] * distance),
    )
    if not _point_allowed(room, point, routing_region, config.allowed_region_buffer_mm):
        return None
    routing_point = (
        round(point[0] - inward[0] * config.connector_length_mm),
        round(point[1] - inward[1] * config.connector_length_mm),
    )
    if not _routing_allowed(routing_point, routing_region):
        return None
    return _Candidate(
        point, routing_point, role, inward, index, length, float(distance), int(width),
    )


def _simplified_boundary_lines(polygon):
    coords = list(polygon.exterior.coords)
    simplified = [coords[0]]
    for index in range(1, len(coords) - 1):
        previous, current, following = simplified[-1], coords[index], coords[index + 1]
        first = (current[0] - previous[0], current[1] - previous[1])
        second = (following[0] - current[0], following[1] - current[1])
        first_length, second_length = math.hypot(*first), math.hypot(*second)
        if first_length < 20.0 or second_length < 20.0:
            continue
        cross_ratio = abs(first[0] * second[1] - first[1] * second[0]) / max(
            first_length * second_length, 1.0,
        )
        if cross_ratio >= math.sin(0.3):
            simplified.append(current)
    simplified.append(simplified[0])
    return tuple(
        LineString((first, second))
        for first, second in zip(simplified, simplified[1:])
        if first != second
    )


def _line_parts(geometry):
    if geometry is None or geometry.is_empty:
        return ()
    if isinstance(geometry, LineString):
        return (geometry,)
    if isinstance(geometry, MultiLineString):
        return tuple(geometry.geoms)
    if hasattr(geometry, "geoms"):
        return tuple(line for part in geometry.geoms for line in _line_parts(part))
    return ()


def _eligible_segments(room_polygon, allowed, minimum_length):
    result = []
    for line in _simplified_boundary_lines(room_polygon):
        if line.length < minimum_length:
            continue
        distance = float(allowed.distance(line))
        expansion = 300.0 if math.isclose(distance, 0.0, abs_tol=100.0) else min(distance * 1.2, 300.0)
        for part in _line_parts(allowed.buffer(expansion, cap_style=2, join_style=3).intersection(line)):
            if part.length < minimum_length:
                continue
            coords = list(part.coords)
            result.append({
                "p0": tuple(map(float, coords[0])),
                "p1": tuple(map(float, coords[-1])),
                "length": float(part.length),
            })
    return result


def _overlap(left, right):
    if left.segment_index != right.segment_index:
        return 0.0
    left_span = (left.distance_along - left.width_mm / 2, left.distance_along + left.width_mm / 2)
    right_span = (right.distance_along - right.width_mm / 2, right.distance_along + right.width_mm / 2)
    return max(0.0, min(left_span[1], right_span[1]) - max(left_span[0], right_span[0]))


def _valid_pair(supply, returned, config):
    if supply is None or returned is None or _overlap(supply, returned) > config.maximum_overlap_mm:
        return False
    if supply.segment_index == returned.segment_index:
        gap = abs(supply.distance_along - returned.distance_along)
        required = (supply.width_mm + returned.width_mm) / 2 + config.minimum_between_mm
        return gap + config.maximum_overlap_mm >= required
    return True


def _at_side(room, segment, index, width, role, start, routing_region, config):
    margin = config.minimum_wall_distance_mm + width / 2
    distance = margin if start else segment["length"] - margin
    return _candidate(room, segment, index, distance, width, role, routing_region, config)


def _at_center(room, segment, index, width, role, routing_region, config):
    return _candidate(room, segment, index, segment["length"] / 2, width, role, routing_region, config)


def _machine_aligned(room, segment, index, routing_region, config, machine, pins, angle):
    if machine is None or not pins or "air_out" not in pins:
        return None
    connector = machine.connector("air_out")
    direction = local_axis_to_world(connector.normal_xy, angle)
    origin = pins["air_out"]
    ray = LineString((origin, (origin[0] + direction[0] * 4000, origin[1] + direction[1] * 4000)))
    hit = LineString((segment["p0"], segment["p1"])).intersection(ray)
    if not isinstance(hit, Point):
        return None
    vector = _unit_and_inward(segment["p0"], segment["p1"], room.polygon)
    if vector is None:
        return None
    unit, _inward, length = vector
    distance = (hit.x - segment["p0"][0]) * unit[0] + (hit.y - segment["p0"][1]) * unit[1]
    margin = config.minimum_wall_distance_mm + config.supply_width_mm / 2
    if distance < margin or distance > length - margin:
        return None
    return _candidate(
        room, segment, index, distance, config.supply_width_mm, "supply",
        routing_region, config,
    )


def _largest_polygon(geometry):
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, Polygon):
        return geometry
    polygons = [part for part in getattr(geometry, "geoms", ()) if isinstance(part, Polygon)]
    return max(polygons, key=lambda item: item.area) if polygons else None


def _cover_region(room, routing_region):
    coverings = getattr(room, "space_coverings", None) or ()
    polygons = [cover for cover, _full in coverings if cover is not None and not cover.is_empty]
    if polygons:
        return unary_union(polygons)
    if routing_region is None or routing_region.is_empty:
        return None
    return room.polygon.intersection(routing_region)


def _wall_options(room, segments, routing_region, config, machine, pins, angle):
    options = []
    for index, segment in enumerate(segments):
        aligned = _machine_aligned(room, segment, index, routing_region, config, machine, pins, angle)
        supply_start = _at_side(room, segment, index, config.supply_width_mm, "supply", True, routing_region, config)
        supply_end = _at_side(room, segment, index, config.supply_width_mm, "supply", False, routing_region, config)
        return_start = _at_side(room, segment, index, config.return_width_mm, "return", True, routing_region, config)
        return_end = _at_side(room, segment, index, config.return_width_mm, "return", False, routing_region, config)
        for kind, pair in (
            ("common_wall_machine_aligned", (aligned, return_start if aligned and aligned.distance_along > segment["length"] / 2 else return_end)),
            ("common_wall", (supply_start, return_end)),
            ("common_wall_alt", (supply_end, return_start)),
        ):
            if _valid_pair(*pair, config):
                options.append((kind, *pair))
    for left_index, left in enumerate(segments):
        supply = _machine_aligned(room, left, left_index, routing_region, config, machine, pins, angle) or _at_center(
            room, left, left_index, config.supply_width_mm, "supply", routing_region, config,
        )
        for right_index in range(left_index + 1, len(segments)):
            right = segments[right_index]
            returned = _at_center(room, right, right_index, config.return_width_mm, "return", routing_region, config)
            if _valid_pair(supply, returned, config):
                options.append(("different_walls", supply, returned))
            alternate_supply = _at_center(room, right, right_index, config.supply_width_mm, "supply", routing_region, config)
            alternate_return = _at_center(room, left, left_index, config.return_width_mm, "return", routing_region, config)
            if _valid_pair(alternate_supply, alternate_return, config):
                options.append(("different_walls_alt", alternate_supply, alternate_return))
    return options


def _options(room, routing_region, config, machine, pins, angle):
    allowed = _cover_region(room, routing_region)
    if allowed is None or allowed.is_empty:
        allowed = room.polygon
    false_ceiling = _largest_polygon(room.polygon.intersection(allowed))
    if false_ceiling is not None and false_ceiling.area >= config.minimum_false_ceiling_area_mm2:
        uncovered = _largest_polygon(room.polygon.difference(allowed))
        if uncovered is not None:
            segments = _eligible_segments(
                uncovered, allowed, config.supply_width_mm + 2 * config.minimum_wall_distance_mm,
            )
            return_point = false_ceiling.representative_point()
            horizontal_return = _Candidate(
                (round(return_point.x), round(return_point.y)),
                (round(return_point.x), round(return_point.y)), "return", (0.0, 0.0),
                -1, 0.0, 0.0, config.return_width_mm, "horizontal_return",
            )
            horizontal = []
            for index, segment in enumerate(segments):
                supply = _at_center(
                    room, segment, index, config.supply_width_mm, "supply", routing_region, config,
                )
                if _valid_pair(supply, horizontal_return, config):
                    horizontal.append(("return_horizontal", supply, horizontal_return))
            if horizontal:
                return horizontal
    segments = _eligible_segments(room.polygon, allowed, config.minimum_wall_length_mm)
    return _wall_options(room, segments, routing_region, config, machine, pins, angle)


def prepare_cli_grilles(
    rooms, routing_region, *, machine=None, machine_pins=None, angle_degrees=0,
    config=CliGrillePlacementConfig(), supply_width_by_room_id=None,
):
    """Select the first valid legacy-compatible supply/return option per Cli room."""
    targets = [room for room in rooms if is_cli_supply_room(room)]
    connections, counts = [], {}
    for room in targets:
        room_config = config
        supply_sizing = None
        if supply_width_by_room_id and id(room) in supply_width_by_room_id:
            supply_sizing = supply_width_by_room_id[id(room)]
            room_config = dataclass_replace(
                config, supply_width_mm=supply_sizing.width_mm,
            )
        base_name = str(getattr(room, "name", "") or "Cli").strip() or "Cli"
        counts[base_name] = counts.get(base_name, 0) + 1
        room_name = base_name if counts[base_name] == 1 else f"{base_name} {counts[base_name]}"
        options = _options(
            room, routing_region, room_config, machine, machine_pins, angle_degrees,
        )
        if not options:
            continue
        placement_type, supply, returned = options[0]
        for candidate in (supply, returned):
            connections.append(CliGrilleConnection(
                f"{room_name} {'Supply' if candidate.role == 'supply' else 'Return'}",
                room_name, candidate.role, candidate.point, candidate.routing_point,
                candidate.width_mm, candidate.inward_normal, placement_type,
                float(room.polygon.area) / 1_000_000.0,
                supply_sizing if candidate.role == "supply" else None,
            ))
    return CliGrillePlan(tuple(connections))
