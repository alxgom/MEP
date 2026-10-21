"""Sampled-boundary medial-axis grid; no path search or route selection.

SciPy's point-site Voronoi diagram approximates the polygon medial axis as
boundary sampling is refined. Same-boundary-segment ridges are discarded.
This is an approximation, not an exact segment-site Voronoi implementation.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Point
from shapely import wkb

from mep_routing.geometry.directions import travel_direction
from mep_routing.geometry.polygons import iter_polygons
from .axis_grid import _edge_allowed


@dataclass(frozen=True)
class MedialTemplate:
    nodes: np.ndarray
    edges: tuple
    sample_count: int


@lru_cache(maxsize=16)
def cached_medial_template(region_wkb, walls_wkb, wall_thickness_mm, sample_spacing_mm):
    """Reuse static geometry across machine poses and installation sessions."""
    result = build_medial_template(wkb.loads(region_wkb), tuple(wkb.loads(p) for p in walls_wkb),
                                  wall_thickness_mm, sample_spacing_mm=sample_spacing_mm)
    result.nodes.setflags(write=False)
    return result


def _records(points, pairs):
    edges = []
    for u, v in sorted(set((min(a, b), max(a, b)) for a, b in pairs if a != b)):
        length = math.dist(points[u], points[v])
        if length > 1e-5:
            edges.append((u, v, length, travel_direction(points[u], points[v])))
    return edges


def _compact_chains(nodes, pairs, allowed, wall_polys, wall_thickness_mm, tolerance):
    """Simplify degree-two chains, retaining junctions, endpoints and cycles.

    Simplified chords become explicit straight graph edges. Validate every
    chord against geometry and walls; retain the original chain on rejection.
    """
    adjacency = {i: set() for i in range(len(nodes))}
    for u, v in pairs:
        adjacency[u].add(v)
        adjacency[v].add(u)
    visited, paths = set(), []
    # Visit branch/end nodes first; the second pass also handles closed cycles.
    starts = sorted(adjacency, key=lambda i: (len(adjacency[i]) == 2, i))
    for start in starts:
        for neighbour in sorted(adjacency[start]):
            if (min(start, neighbour), max(start, neighbour)) in visited:
                continue
            chain = [start, neighbour]
            visited.add((min(start, neighbour), max(start, neighbour)))
            while len(adjacency[chain[-1]]) == 2 and chain[-1] != start:
                following = next(i for i in adjacency[chain[-1]] if i != chain[-2])
                visited.add((min(chain[-1], following), max(chain[-1], following)))
                chain.append(following)
            paths.append([tuple(map(float, nodes[i])) for i in chain])
    points, indices, compact_pairs = [], {}, []
    wall_bounds = [p.bounds for p in wall_polys]
    for path in paths:
        candidate = list(LineString(path).simplify(tolerance).coords)
        if not all(_edge_allowed(LineString((a, b)), allowed, wall_polys, wall_bounds, wall_thickness_mm)
                   for a, b in zip(candidate, candidate[1:])):
            candidate = path
        chain = []
        for point in candidate:
            if point not in indices:
                indices[point] = len(points)
                points.append(point)
            chain.append(indices[point])
        compact_pairs.extend(zip(chain, chain[1:]))
    result = np.asarray(points, dtype=np.float32).reshape((-1, 2))
    edges = _records(result, compact_pairs)
    return result, tuple(edges)


def build_medial_template(allowed_region, wall_polys=(), wall_thickness_mm=150, *, sample_spacing_mm=200):
    """Build an immutable, reusable skeleton for all connected polygon parts."""
    if not math.isfinite(sample_spacing_mm) or sample_spacing_mm <= 0:
        raise ValueError("medial boundary sample spacing must be finite and positive")
    empty = MedialTemplate(np.empty((0, 2), dtype=np.float32), (), 0)
    if allowed_region is None or allowed_region.is_empty:
        return empty
    samples, features = [], []
    feature = 0
    # Remove redundant collinear polygon vertices without moving the boundary.
    for polygon in iter_polygons(allowed_region.simplify(0)):
        for ring in (polygon.exterior, *polygon.interiors):
            coords = list(ring.coords)
            for start, end in zip(coords, coords[1:]):
                count = max(1, math.ceil(math.dist(start, end) / sample_spacing_mm))
                for fraction in np.arange(count) / count:
                    samples.append((start[0] + fraction * (end[0] - start[0]), start[1] + fraction * (end[1] - start[1])))
                    features.append(feature)
                feature += 1
    if len(samples) < 4:
        return empty
    sites, unique = np.unique(np.asarray(samples), axis=0, return_index=True)
    features = np.asarray(features)[unique]
    diagram = Voronoi(sites)
    nodes, pairs, indices = [], [], {}
    wall_bounds = [wall.bounds for wall in wall_polys]
    for generators, ridge in zip(diagram.ridge_points, diagram.ridge_vertices):
        if len(ridge) != 2 or min(ridge) < 0 or features[generators[0]] == features[generators[1]]:
            continue
        points = diagram.vertices[ridge]
        if not _edge_allowed(LineString(points), allowed_region, wall_polys, wall_bounds, wall_thickness_mm):
            continue
        edge = []
        for point in points:
            # Float32 is the common runtime coordinate representation.
            key = tuple(np.asarray(point, dtype=np.float32))
            if key not in indices:
                indices[key] = len(nodes)
                nodes.append(key)
            edge.append(indices[key])
        pairs.append(tuple(edge))
    if not pairs:
        return MedialTemplate(empty.nodes, (), len(sites))
    points, edges = _compact_chains(np.asarray(nodes), pairs, allowed_region, wall_polys,
                                   wall_thickness_mm, sample_spacing_mm * 0.05)
    return MedialTemplate(points, edges, len(sites))


def attach_medial_points(template, required_points, allowed_region, wall_polys=(), wall_thickness_mm=150, *, machine_polygon=None):
    """Join exact endpoints to nearby visible skeleton segments, splitting them.

    Attachments never bridge a polygon hole or disconnected free-space parts.
    Multiple local connections retain alternatives when the machine cuts a branch.
    """
    nodes = [tuple(float(x) for x in point) for point in template.nodes]
    indices = {point: i for i, point in enumerate(nodes)}
    splits = {i: [(0.0, edge[0]), (1.0, edge[1])] for i, edge in enumerate(template.edges)}
    connections = []
    wall_bounds = [wall.bounds for wall in wall_polys]
    starts = np.asarray([nodes[u] for u, _v, _l, _d in template.edges]).reshape((-1, 2))
    ends = np.asarray([nodes[v] for _u, v, _l, _d in template.edges]).reshape((-1, 2))
    delta = ends - starts
    length_sq = np.sum(delta * delta, axis=1)

    def index(point):
        key = tuple(float(x) for x in np.asarray(point, dtype=np.float32))
        if key not in indices:
            indices[key] = len(nodes)
            nodes.append(key)
        return indices[key]

    for raw_point in sorted(set(tuple(map(float, p)) for p in required_points)):
        point = np.asarray(raw_point)
        if not allowed_region.covers(Point(raw_point)):
            continue
        source = index(point)
        if not len(starts):
            continue
        fractions = np.clip(np.sum((point - starts) * delta, axis=1) / length_sq, 0, 1)
        projections = starts + fractions[:, None] * delta
        distances = np.sum((projections - point) ** 2, axis=1)
        attached = set()
        # Bound per-endpoint attachment effort; do not connect to distant rooms.
        for edge_index in np.argsort(distances, kind="stable")[:24]:
            projection = projections[edge_index]
            line = LineString((point, projection))
            if not _edge_allowed(line, allowed_region, wall_polys, wall_bounds, wall_thickness_mm):
                continue
            if machine_polygon is not None and line.intersection(machine_polygon).length > 1.0:
                continue
            target = index(projection)
            splits[int(edge_index)].append((float(fractions[edge_index]), target))
            if target != source:
                connections.append((source, target))
            attached.add(target)
            if len(attached) == 3:
                break
    for values in splits.values():
        ordered = sorted(set(values))
        connections.extend((left[1], right[1]) for left, right in zip(ordered, ordered[1:]))
    points = np.asarray(nodes, dtype=np.float32).reshape((-1, 2))
    return points, _records(points, connections)
