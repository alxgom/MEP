"""Plan-view CLI grille supports and paired placement options."""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dataclass_replace
from functools import lru_cache
import math

from shapely.geometry import LineString, Point, Polygon
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
    supply_routing_clearance_factor: float = 0.6


@dataclass(frozen=True)
class CliGrilleConnection:
    route_name: str
    room_name: str
    role: str
    grille_point: tuple[float, float]
    routing_point: tuple[int, int]
    width_mm: int
    inward_normal: tuple[float, float]
    placement_type: str = "wall"
    room_area_m2: float = 0.0
    sizing: object | None = None


@dataclass(frozen=True)
class CliGrillePlan:
    connections: tuple[CliGrilleConnection, ...]
    pairs_by_supply: dict[str, tuple[tuple[CliGrilleConnection, CliGrilleConnection], ...]] = field(
        default_factory=dict,
    )
    missing_rooms: tuple[str, ...] = ()

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
    point: tuple[float, float]
    routing_point: tuple[int, int]
    role: str
    inward_normal: tuple[float, float]
    segment_index: int
    segment_length: float
    distance_along: float
    width_mm: int
    placement_type: str = "wall"


def _routing_allowed(point, routing_region):
    return routing_region is None or routing_region.covers(Point(point))


@lru_cache(maxsize=32)
def _supply_routing_region(region, clearance):
    return None if region is None else region.buffer(-clearance)


def _candidate(room, segment, index, distance, width, role, routing_region, config):
    unit, inward, length = segment["geometry"]
    margin = config.minimum_wall_distance_mm + width / 2
    if not margin - 1e-7 <= distance <= length - margin + 1e-7:
        return None
    point = (
        segment["p0"][0] + unit[0] * distance,
        segment["p0"][1] + unit[1] * distance,
    )
    if room.polygon.distance(Point(point)) > 1e-7:
        return None
    routing_point = (
        round(point[0] - inward[0] * config.connector_length_mm),
        round(point[1] - inward[1] * config.connector_length_mm),
    )
    if role == "supply":
        routing_region = _supply_routing_region(
            routing_region, config.connector_length_mm * config.supply_routing_clearance_factor,
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


@lru_cache(maxsize=128)
def _eligible_segments(room_polygon, allowed, minimum_length, reach_limit):
    result = []
    inward_sign = 1 if room_polygon.exterior.is_ccw else -1
    for line in _simplified_boundary_lines(room_polygon):
        if line.length < minimum_length:
            continue
        distance = float(allowed.distance(line))
        expansion = reach_limit if distance <= 100.0 else min(distance * 1.2, reach_limit)
        part = allowed.buffer(expansion, cap_style=2, join_style=3).intersection(line)
        if not isinstance(part, LineString) or part.is_empty or part.length < minimum_length:
            continue
        first, last = part.coords[0], part.coords[-1]
        if line.project(Point(first)) > line.project(Point(last)):
            first, last = last, first
        dx, dy = last[0] - first[0], last[1] - first[1]
        length = math.hypot(dx, dy)
        unit = (dx / length, dy / length)
        inward = (-inward_sign * unit[1], inward_sign * unit[0])
        result.append({"p0": first, "p1": last, "length": length, "geometry": (unit, inward, length)})
    return tuple(result)


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
    unit, _inward, length = segment["geometry"]
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
    return routing_region


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
    covered = room.polygon.intersection(allowed)
    if covered.area >= config.minimum_false_ceiling_area_mm2:
        uncovered = _largest_polygon(room.polygon.difference(allowed))
        if uncovered is not None:
            segments = _eligible_segments(
                uncovered, allowed, config.supply_width_mm + 2 * config.minimum_wall_distance_mm,
                config.allowed_region_buffer_mm,
            )
            false_ceiling = _largest_polygon(covered if routing_region is None else covered.intersection(routing_region))
            if false_ceiling is None or false_ceiling.area <= config.minimum_false_ceiling_area_mm2:
                return []
            return_point = false_ceiling.centroid
            if not false_ceiling.covers(return_point):
                return_point = false_ceiling.representative_point()
            supply_region = allowed if routing_region is None else allowed.intersection(routing_region)
            horizontal_return = _Candidate(
                (round(return_point.x), round(return_point.y)),
                (round(return_point.x), round(return_point.y)), "return", (0.0, 0.0),
                -1, 0.0, 0.0, config.return_width_mm, "horizontal_return",
            )
            horizontal = []
            for index, segment in enumerate(segments):
                supply = _at_center(
                    room, segment, index, config.supply_width_mm, "supply", supply_region, config,
                )
                if _valid_pair(supply, horizontal_return, config):
                    horizontal.append(("return_horizontal", supply, horizontal_return))
            return horizontal
        if getattr(room, "space_coverings", ()):
            return []
    segments = _eligible_segments(
        room.polygon, allowed, config.minimum_wall_length_mm, config.allowed_region_buffer_mm,
    )
    return _wall_options(room, segments, routing_region, config, machine, pins, angle)


def prepare_cli_grilles(
    rooms, routing_region, *, machine=None, machine_pins=None, angle_degrees=0,
    config=CliGrillePlacementConfig(), supply_width_by_room_id=None,
):
    """Prepare one fixed pair and the distinct routable supply choices per CLI room."""
    targets = [room for room in rooms if is_cli_supply_room(room)]
    connections, counts, pairs_by_supply, missing_rooms = [], {}, {}, []
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
            missing_rooms.append(room_name)
            continue
        pairs = {}
        for placement_type, supply, returned in options:
            key = (supply.point, supply.routing_point, supply.inward_normal)
            if key in pairs:
                continue
            pairs[key] = tuple(
                CliGrilleConnection(
                    f"{room_name} {'Supply' if candidate.role == 'supply' else 'Return'}",
                    room_name, candidate.role, candidate.point, candidate.routing_point,
                    candidate.width_mm, candidate.inward_normal, placement_type,
                    float(room.polygon.area) / 1_000_000.0,
                    supply_sizing if candidate.role == "supply" else None,
                )
                for candidate in (supply, returned)
            )
        room_pairs = tuple(pairs.values())
        connections.extend(room_pairs[0])
        pairs_by_supply[room_pairs[0][0].route_name] = room_pairs
    return CliGrillePlan(tuple(connections), pairs_by_supply, tuple(missing_rooms))
