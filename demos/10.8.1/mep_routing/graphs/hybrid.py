"""Join Hanan and Voronoi grids with bounded local links and shared junctions."""

import math

import numpy as np
from shapely import STRtree, linestrings
from shapely.geometry import LineString

from mep_routing.geometry.directions import travel_direction
from .axis_grid import _edge_allowed


def _intersection_points(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Point":
        yield geometry.coords[0]
    elif geometry.geom_type == "LineString":
        yield geometry.coords[0]
        yield geometry.coords[-1]
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _intersection_points(part)


def join_hanan_voronoi(base_nodes, base_edges, medial_nodes, medial_edges, allowed_region,
                       wall_polys=(), wall_thickness_mm=150, *, max_link_distance_mm=600):
    """Keep both grids, attaching each Voronoi node in up to four directions.

    The first legal Hanan-edge projection in each cardinal direction is used
    within the radius. Existing crossings also become shared nodes. No axis is
    extended beyond the local connector. Both input grids remain unchanged.
    """
    if not math.isfinite(max_link_distance_mm) or max_link_distance_mm <= 0:
        raise ValueError("hybrid link distance must be finite and positive")
    points = [tuple(map(float, p)) for p in base_nodes]
    indices = {p: i for i, p in enumerate(points)}

    def index(point):
        key = tuple(map(float, np.asarray(point, dtype=np.float32)))
        if key not in indices:
            indices[key] = len(points)
            points.append(key)
        return indices[key]

    pairs = [(u, v) for u, v, *_ in base_edges]
    base_count = len(pairs)
    wall_bounds = [wall.bounds for wall in wall_polys]

    def add_segment(a, b):
        if math.dist(a, b) < 1e-4:
            return False
        if not _edge_allowed(LineString((a, b)), allowed_region, wall_polys, wall_bounds, wall_thickness_mm):
            return False
        pairs.append((index(a), index(b)))
        return True

    for u, v, *_ in medial_edges:
        add_segment(medial_nodes[u], medial_nodes[v])

    # Numeric projections to the original Hanan edges avoid global axis growth.
    starts = np.asarray([base_nodes[u] for u, v, *_ in base_edges]).reshape((-1, 2))
    ends = np.asarray([base_nodes[v] for u, v, *_ in base_edges]).reshape((-1, 2))
    vertical = np.abs(starts[:, 0] - ends[:, 0]) < 1e-4
    horizontal = np.abs(starts[:, 1] - ends[:, 1]) < 1e-4
    lower, upper = np.minimum(starts, ends), np.maximum(starts, ends)
    links, linked_nodes = 0, 0
    for point in medial_nodes:
        candidates = {direction: [] for direction in ("E", "W", "N", "S")}
        for edge_index in np.flatnonzero(vertical & (lower[:, 1] <= point[1]) & (point[1] <= upper[:, 1])):
            target = (float(starts[edge_index, 0]), float(point[1]))
            distance = abs(target[0] - point[0])
            if 1e-4 < distance <= max_link_distance_mm:
                candidates["E" if target[0] > point[0] else "W"].append((float(distance), target))
        for edge_index in np.flatnonzero(horizontal & (lower[:, 0] <= point[0]) & (point[0] <= upper[:, 0])):
            target = (float(point[0]), float(starts[edge_index, 1]))
            distance = abs(target[1] - point[1])
            if 1e-4 < distance <= max_link_distance_mm:
                candidates["N" if target[1] > point[1] else "S"].append((float(distance), target))
        attached = False
        for values in candidates.values():
            for _distance, target in sorted(set(values)):
                if add_segment(point, target):
                    links += 1
                    attached = True
                    break
        linked_nodes += attached

    # Query only added segments against the base and each other. The original
    # Hanan topology already has explicit crossings; it need not be rebuilt.
    if not pairs:
        return np.asarray(points, dtype=np.float32).reshape((-1, 2)), [], {"links": links, "linked_nodes": linked_nodes}
    segments = linestrings(np.asarray([[points[u], points[v]] for u, v in pairs]))
    splits = [{u, v} for u, v in pairs]
    base_tree, added_tree = STRtree(segments[:base_count]), STRtree(segments[base_count:])
    for first in range(base_count, len(pairs)):
        others = list(map(int, base_tree.query(segments[first], predicate="intersects")))
        others.extend(base_count + int(i) for i in added_tree.query(segments[first], predicate="intersects")
                      if base_count + int(i) > first)
        for second in sorted(others):
            for point in _intersection_points(segments[first].intersection(segments[second])):
                junction = index(point)
                splits[first].add(junction)
                splits[second].add(junction)
    result_pairs = set()
    for (u, v), junctions in zip(pairs, splits):
        start, end = np.asarray(points[u]), np.asarray(points[v])
        delta = end - start
        ordered = sorted(junctions, key=lambda i: (float(np.dot(np.asarray(points[i]) - start, delta)), i))
        result_pairs.update((min(a, b), max(a, b)) for a, b in zip(ordered, ordered[1:]) if a != b)
    nodes = np.asarray(points, dtype=np.float32).reshape((-1, 2))
    edges = [(u, v, math.dist(nodes[u], nodes[v]), travel_direction(nodes[u], nodes[v]))
             for u, v in sorted(result_pairs) if math.dist(nodes[u], nodes[v]) > 1e-4]
    return nodes, edges, {"links": links, "linked_nodes": linked_nodes}
