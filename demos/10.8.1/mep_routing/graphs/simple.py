"""Installation-neutral port of routing-core's simple orthogonal grid."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Iterable

import numpy as np
from shapely.geometry import LineString, Point

from mep_routing.geometry import largest_polygon

from .axes import merge_close_values

Edge = tuple[int, int, float, str]


@dataclass(frozen=True)
class SimpleGraphInputs:
    allowed_region: object
    required_points: tuple[tuple[float, float], ...]
    obstacles: tuple[object, ...] = ()
    epsilon_mm: float = 200.0
    axis_clearance_mm: float = 100.0
    blocked_edge_weight: float = 1e6
    boundary_inset_mm: float | None = None
    boundary_cluster_mm: float = 300.0
    boundary_inner_clearance_mm: float = 110.0
    scaffold_spacing_mm: float | None = None


@dataclass(frozen=True)
class SimpleGraphResult:
    nodes: np.ndarray
    edges: list[Edge]
    axes_x: list[int]
    axes_y: list[int]
    required_points: list[tuple[float, float]]
    axes_ms: float
    nodes_ms: float
    edges_ms: float


def _boundary_axes(inputs: SimpleGraphInputs) -> tuple[list[int], list[int]]:
    poly = largest_polygon(inputs.allowed_region)
    if poly is None:
        return [], []
    points = list(poly.exterior.coords)[:-1]
    if len(points) < 3:
        return [], []
    inset = inputs.epsilon_mm if inputs.boundary_inset_mm is None else inputs.boundary_inset_mm
    inward = []
    for index, point in enumerate(points):
        previous = np.asarray(points[index - 1], dtype=np.float64)
        current = np.asarray(point, dtype=np.float64)
        following = np.asarray(points[(index + 1) % len(points)], dtype=np.float64)
        first, second = previous - current, following - current
        first_norm, second_norm = float(np.linalg.norm(first)), float(np.linalg.norm(second))
        if first_norm <= 1e-7 or second_norm <= 1e-7:
            continue
        angle = np.degrees(np.arccos(np.clip(np.dot(first, second) / (first_norm * second_norm), -1.0, 1.0)))
        if angle >= 170.0:
            continue
        direction = first / first_norm + second / second_norm
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-7:
            continue
        direction /= direction_norm
        candidate = current + direction * inset
        if not poly.contains(Point(float(candidate[0]), float(candidate[1]))):
            candidate = current - direction * inset
        inner = poly.buffer(-inputs.boundary_inner_clearance_mm).convex_hull
        if not inner.is_empty and inner.contains(Point(float(candidate[0]), float(candidate[1]))):
            inward.append(candidate)
    if not inward:
        return [], []
    coordinates = np.asarray(inward, dtype=np.float64)
    used = np.zeros(len(coordinates), dtype=bool)
    clustered = []
    for index, point in enumerate(coordinates):
        if used[index]:
            continue
        group = np.linalg.norm(coordinates - point, axis=1) < inputs.boundary_cluster_mm
        used[group] = True
        clustered.append(coordinates[group].mean(axis=0))

    def spaced(values: Iterable[float]) -> list[int]:
        result = []
        for value in sorted(float(value) for value in values):
            if not result or value - result[-1] >= inputs.epsilon_mm:
                result.append(round(value))
        return result

    return spaced(point[0] for point in clustered), spaced(point[1] for point in clustered)


def build_simple_grid(inputs: SimpleGraphInputs) -> SimpleGraphResult:
    """Build a sparse axis grid, retaining blocked edges at the configured cost."""
    started = perf_counter()
    allowed = inputs.allowed_region
    required_points = [(round(float(x)), round(float(y))) for x, y in inputs.required_points]
    if allowed is None or allowed.is_empty or not required_points:
        return SimpleGraphResult(np.empty((0, 2), dtype=np.float32), [], [], [], required_points, 0.0, 0.0, 0.0)
    axes_x, axes_y = {p[0] for p in required_points}, {p[1] for p in required_points}
    boundary_x, boundary_y = _boundary_axes(inputs)
    axes_x.update(boundary_x)
    axes_y.update(boundary_y)
    if inputs.scaffold_spacing_mm and inputs.scaffold_spacing_mm > 0:
        min_x, min_y, max_x, max_y = allowed.bounds
        spacing = float(inputs.scaffold_spacing_mm)
        axes_x.update(range(math.ceil(min_x / spacing) * round(spacing), math.floor(max_x / spacing) * round(spacing) + 1, round(spacing)))
        axes_y.update(range(math.ceil(min_y / spacing) * round(spacing), math.floor(max_y / spacing) * round(spacing) + 1, round(spacing)))
    obstacles = tuple(o for o in inputs.obstacles if o is not None and not o.is_empty)
    for obstacle in obstacles:
        min_x, min_y, max_x, max_y = obstacle.buffer(inputs.axis_clearance_mm, join_style=2).bounds
        axes_x.update((round(float(min_x)), round(float(max_x))))
        axes_y.update((round(float(min_y)), round(float(max_y))))
    xs = merge_close_values(axes_x, inputs.axis_clearance_mm, preserve_values={p[0] for p in required_points}, priority_values=boundary_x)
    ys = merge_close_values(axes_y, inputs.axis_clearance_mm, preserve_values={p[1] for p in required_points}, priority_values=boundary_y)
    axes_ms = (perf_counter() - started) * 1000.0
    required_set, node_map, nodes = set(required_points), {}, []
    for y in ys:
        for x in xs:
            point, geometry = (x, y), Point(float(x), float(y))
            if point not in required_set and (not allowed.contains(geometry) or any(o.contains(geometry) for o in obstacles)):
                continue
            node_map[point] = len(nodes)
            nodes.append(point)
    nodes_array = np.asarray(nodes, dtype=np.float32).reshape((-1, 2))
    nodes_ms = (perf_counter() - started) * 1000.0 - axes_ms
    edges: list[Edge] = []
    for y in ys:
        for x1, x2 in zip(xs, xs[1:]):
            source, target = node_map.get((x1, y)), node_map.get((x2, y))
            if source is None or target is None:
                continue
            line = LineString([(float(x1), float(y)), (float(x2), float(y))])
            blocked = not line.covered_by(allowed) or any(line.intersects(o) for o in obstacles)
            edges.append((source, target, inputs.blocked_edge_weight if blocked else float(abs(x2 - x1)), "E"))
    for x in xs:
        for y1, y2 in zip(ys, ys[1:]):
            source, target = node_map.get((x, y1)), node_map.get((x, y2))
            if source is None or target is None:
                continue
            line = LineString([(float(x), float(y1)), (float(x), float(y2))])
            blocked = not line.covered_by(allowed) or any(line.intersects(o) for o in obstacles)
            edges.append((source, target, inputs.blocked_edge_weight if blocked else float(abs(y2 - y1)), "N"))
    edges_ms = (perf_counter() - started) * 1000.0 - axes_ms - nodes_ms
    return SimpleGraphResult(nodes_array, edges, xs, ys, required_points, axes_ms, nodes_ms, edges_ms)
