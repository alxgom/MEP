from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point
from shapely.prepared import prep


def filter_edges_against_walls(raw_edges, nodes, wall_polys, wall_thickness_mm):
    """Keep regular-grid edges that do not cross a wall by more than its thickness."""
    wall_bounds = [wall.bounds for wall in wall_polys]
    valid_edges = []
    for u, v, length, direction in raw_edges:
        p1, p2 = nodes[u], nodes[v]
        min_x, min_y = float(min(p1[0], p2[0])), float(min(p1[1], p2[1]))
        max_x, max_y = float(max(p1[0], p2[0])), float(max(p1[1], p2[1]))
        line = None
        blocked = False
        for wall, (wall_min_x, wall_min_y, wall_max_x, wall_max_y) in zip(wall_polys, wall_bounds):
            if not (max_x >= wall_min_x - 1.0 and min_x <= wall_max_x + 1.0 and max_y >= wall_min_y - 1.0 and min_y <= wall_max_y + 1.0):
                continue
            if line is None:
                line = LineString([(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))])
            intersection = line.intersection(wall)
            if not intersection.is_empty and intersection.length > wall_thickness_mm + 1:
                blocked = True
                break
        if not blocked:
            valid_edges.append((u, v, length, direction))
    return valid_edges


def build_regular_grid(allowed_region, node_region, wall_polys, grid_spacing_mm, wall_thickness_mm):
    """Build regular-grid nodes and valid axis-aligned edges in millimetres."""
    if allowed_region is None or node_region is None:
        return np.empty((0, 2), dtype=np.int32), []

    min_x, min_y, max_x, max_y = allowed_region.bounds
    xs = np.arange(int(min_x // grid_spacing_mm) * grid_spacing_mm, int(max_x // grid_spacing_mm + 1) * grid_spacing_mm + 1, grid_spacing_mm, dtype=np.int32)
    ys = np.arange(int(min_y // grid_spacing_mm) * grid_spacing_mm, int(max_y // grid_spacing_mm + 1) * grid_spacing_mm + 1, grid_spacing_mm, dtype=np.int32)
    x_values, y_values = np.meshgrid(xs, ys)
    candidates = np.column_stack([x_values.ravel(), y_values.ravel()]).astype(np.int32)
    prepared_region = prep(node_region)
    nodes = candidates[np.array([prepared_region.contains(Point(int(x), int(y))) for x, y in candidates], dtype=bool)]

    node_indices = {(int(x), int(y)): index for index, (x, y) in enumerate(nodes)}
    raw_edges = []
    for index, (x, y) in enumerate(nodes):
        east = int(x) + grid_spacing_mm, int(y)
        north = int(x), int(y) + grid_spacing_mm
        if east in node_indices:
            raw_edges.append((index, node_indices[east], grid_spacing_mm, "E"))
        if north in node_indices:
            raw_edges.append((index, node_indices[north], grid_spacing_mm, "N"))

    return nodes, filter_edges_against_walls(raw_edges, nodes, wall_polys, wall_thickness_mm)
