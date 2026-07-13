from __future__ import annotations

import numpy as np
from time import perf_counter
from shapely.geometry import LineString, Point
from shapely.prepared import prep


def _edge_allowed(line, allowed_region, wall_polys, wall_bounds, wall_thickness_mm):
    if allowed_region is None or not line.covered_by(allowed_region):
        return False
    min_x, min_y, max_x, max_y = line.bounds
    for wall, (wall_min_x, wall_min_y, wall_max_x, wall_max_y) in zip(wall_polys, wall_bounds):
        if not (max_x >= wall_min_x - 1.0 and min_x <= wall_max_x + 1.0 and max_y >= wall_min_y - 1.0 and min_y <= wall_max_y + 1.0):
            continue
        intersection = line.intersection(wall)
        if not intersection.is_empty and intersection.length > wall_thickness_mm + 1:
            return False
    return True


def _connect_isolated_required_nodes(nodes, raw_edges, required_points, allowed_region):
    if len(nodes) == 0:
        return raw_edges
    existing = {(min(u, v), max(u, v)) for u, v, _length, _direction in raw_edges}
    degree = {}
    for u, v, _length, _direction in raw_edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1

    by_x, by_y = {}, {}
    for index, (x, y) in enumerate(nodes):
        by_x.setdefault(round(float(x)), []).append(index)
        by_y.setdefault(round(float(y)), []).append(index)

    for point_x, point_y in required_points:
        matches = np.where((np.abs(nodes[:, 0] - point_x) < 1.0) & (np.abs(nodes[:, 1] - point_y) < 1.0))[0]
        if len(matches) == 0:
            continue
        source = int(matches[0])
        if degree.get(source, 0) > 0:
            continue
        candidates = set(by_x.get(round(float(point_x)), [])) | set(by_y.get(round(float(point_y)), []))
        candidates.discard(source)
        for target in sorted(candidates, key=lambda index: float(np.hypot(nodes[index, 0] - point_x, nodes[index, 1] - point_y)))[:12]:
            edge_key = min(source, target), max(source, target)
            if edge_key in existing:
                continue
            line = LineString([(float(nodes[source, 0]), float(nodes[source, 1])), (float(nodes[target, 0]), float(nodes[target, 1]))])
            if allowed_region is None or not line.covered_by(allowed_region):
                continue
            length = float(np.hypot(nodes[target, 0] - nodes[source, 0], nodes[target, 1] - nodes[source, 1]))
            if length < 1.0:
                continue
            direction = "E" if abs(nodes[target, 0] - nodes[source, 0]) > abs(nodes[target, 1] - nodes[source, 1]) and nodes[target, 0] > nodes[source, 0] else "W"
            if abs(nodes[target, 0] - nodes[source, 0]) <= abs(nodes[target, 1] - nodes[source, 1]):
                direction = "N" if nodes[target, 1] > nodes[source, 1] else "S"
            raw_edges.append((source, target, length, direction))
            existing.add(edge_key)
            degree[source] = degree.get(source, 0) + 1
            degree[target] = degree.get(target, 0) + 1
            break
    return raw_edges


def build_axis_grid(xs, ys, allowed_region, node_region, wall_polys, wall_thickness_mm, required_points=()):
    """Build a visibility-filtered axis grid from precomputed x/y axes."""
    start = perf_counter()
    node_map = {}
    nodes = []
    prepared_region = prep(node_region)
    for y in ys:
        for x in xs:
            if prepared_region.contains(Point(float(x), float(y))):
                node_map[(x, y)] = len(nodes)
                nodes.append((x, y))
    if not nodes:
        return np.empty((0, 2), dtype=np.float32), [], (0.0, 0.0)

    nodes_array = np.array(nodes, dtype=np.float32)
    nodes_ms = (perf_counter() - start) * 1000.0
    wall_bounds = [wall.bounds for wall in wall_polys]
    raw_edges = []
    for y in ys:
        row = [x for x in xs if (x, y) in node_map]
        for x1, x2 in zip(row, row[1:]):
            line = LineString([(float(x1), float(y)), (float(x2), float(y))])
            if _edge_allowed(line, allowed_region, wall_polys, wall_bounds, wall_thickness_mm):
                raw_edges.append((node_map[(x1, y)], node_map[(x2, y)], float(abs(x2 - x1)), "E"))
    for x in xs:
        column = [y for y in ys if (x, y) in node_map]
        for y1, y2 in zip(column, column[1:]):
            line = LineString([(float(x), float(y1)), (float(x), float(y2))])
            if _edge_allowed(line, allowed_region, wall_polys, wall_bounds, wall_thickness_mm):
                raw_edges.append((node_map[(x, y1)], node_map[(x, y2)], float(abs(y2 - y1)), "N"))
    raw_edges = _connect_isolated_required_nodes(nodes_array, raw_edges, required_points, allowed_region)
    edges_ms = (perf_counter() - start) * 1000.0 - nodes_ms
    return nodes_array, raw_edges, (nodes_ms, edges_ms)
