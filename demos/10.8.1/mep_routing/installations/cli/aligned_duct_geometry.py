"""Experimental face-aligned polygons over a sized Cli supply skeleton."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

from shapely.geometry import MultiPoint, Polygon


@dataclass(frozen=True)
class CliAlignedDuctProjection:
    conducto_polygons: tuple[Polygon, ...]
    junction_polygons: tuple[Polygon, ...]
    alignment_states: tuple[int, ...]
    event_count: int
    elapsed_ms: float

    @property
    def polygons(self):
        return self.conducto_polygons + self.junction_polygons


def _key(point):
    return round(float(point[0]), 6), round(float(point[1]), 6)


def _unit(start, end):
    dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
    length = math.hypot(dx, dy)
    return (dx / length, dy / length) if length > 1e-7 else (0.0, 0.0)


def _flow(segment):
    return float(getattr(segment.sizing, "airflow_m3h", 0.0) or segment.width_mm)


def _is_collinear(incoming, outgoing, tolerance=1e-7):
    return abs(incoming[0] * outgoing[1] - incoming[1] * outgoing[0]) <= tolerance


def _dominant_state(incoming, dominant, others):
    cross = incoming[0] * dominant[1] - incoming[1] * dominant[0]
    if abs(cross) > 1e-7:
        return -1 if cross > 0 else 1
    if not others:
        return 0
    secondary = max(others, key=lambda item: item[0])
    side = incoming[0] * secondary[1][1] - incoming[1] * secondary[1][0]
    return -1 if side > 1e-7 else 1 if side < -1e-7 else 0


def _orient_tree(segments):
    incidence = {}
    for index, segment in enumerate(segments):
        incidence.setdefault(_key(segment.start), []).append(index)
        incidence.setdefault(_key(segment.end), []).append(index)
    root = next((index for index, item in enumerate(segments) if item.kind == "machine_stub"), 0)
    root_point = _key(segments[root].start)
    oriented, children_at = {}, {}
    pending = [(root_point, None)]
    visited_edges = set()
    while pending:
        point, parent_edge = pending.pop(0)
        outgoing = []
        for edge_index in incidence.get(point, ()):
            if edge_index in visited_edges:
                continue
            visited_edges.add(edge_index)
            segment = segments[edge_index]
            start_key, end_key = _key(segment.start), _key(segment.end)
            other = end_key if start_key == point else start_key
            oriented[edge_index] = (point, other)
            outgoing.append(edge_index)
            pending.append((other, edge_index))
        if parent_edge is not None:
            children_at[parent_edge] = tuple(outgoing)
    return root, oriented, children_at, incidence


def _alignment_states(segments, root, oriented, children_at):
    states = [0] * len(segments)
    events = 0
    pending = [root]
    while pending:
        parent_index = pending.pop(0)
        children = children_at.get(parent_index, ())
        if not children:
            continue
        p0, p1 = oriented[parent_index]
        incoming = _unit(p0, p1)
        child_data = []
        for child_index in children:
            c0, c1 = oriented[child_index]
            direction = _unit(c0, c1)
            continuity = max(0.0, incoming[0] * direction[0] + incoming[1] * direction[1])
            child_data.append((_flow(segments[child_index]) * (1.0 + continuity), direction, child_index))
        if len(children) == 1:
            _score, direction, child_index = child_data[0]
            same_section = segments[parent_index].width_mm == segments[child_index].width_mm
            if _is_collinear(incoming, direction):
                states[child_index] = states[parent_index]
                if not same_section:
                    events += 1
            else:
                states[child_index] = 0
                events += 1
        else:
            events += 1
            dominant = max(child_data, key=lambda item: (item[0], -item[2]))
            other_data = [(score, direction) for score, direction, index in child_data if index != dominant[2]]
            for _score, _direction, child_index in child_data:
                states[child_index] = (
                    _dominant_state(incoming, dominant[1], other_data)
                    if child_index == dominant[2] else 0
                )
        pending.extend(children)
    return tuple(states), events


def _face(segment, point, state, oriented_line):
    direction = _unit(*oriented_line)
    normal = (-direction[1], direction[0])
    offset = float(state) * segment.width_mm / 2.0
    center = (float(point[0]) + normal[0] * offset, float(point[1]) + normal[1] * offset)
    half = segment.width_mm / 2.0
    return (
        (center[0] + normal[0] * half, center[1] + normal[1] * half),
        (center[0] - normal[0] * half, center[1] - normal[1] * half),
    )


def _conducto(segment, state, oriented_line):
    start_face = _face(segment, segment.start, state, oriented_line)
    end_face = _face(segment, segment.end, state, oriented_line)
    polygon = Polygon((start_face[0], end_face[0], end_face[1], start_face[1]))
    return polygon if polygon.is_valid and polygon.area > 0 else None


def _junctions(segments, states, incidence, oriented):
    result = []
    for point, edge_indices in sorted(incidence.items()):
        if len(edge_indices) < 2:
            continue
        points = [
            corner
            for index in edge_indices
            for corner in _face(segments[index], point, states[index], oriented[index])
        ]
        hull = MultiPoint(points).convex_hull
        if isinstance(hull, Polygon) and hull.area > 0:
            result.append(hull)
    return tuple(result)


def build_aligned_duct_polygon_projection(segments):
    """Build one deterministic flow-aware face-aligned projection."""
    started = perf_counter()
    segments = tuple(segments)
    if not segments:
        return CliAlignedDuctProjection((), (), (), 0, 0.0)
    root, oriented, children_at, incidence = _orient_tree(segments)
    states, event_count = _alignment_states(segments, root, oriented, children_at)
    conductos = tuple(
        polygon for index, (segment, state) in enumerate(zip(segments, states))
        if (polygon := _conducto(segment, state, oriented[index])) is not None
    )
    return CliAlignedDuctProjection(
        conductos, _junctions(segments, states, incidence, oriented), states, event_count,
        (perf_counter() - started) * 1000.0,
    )
