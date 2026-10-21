"""Figure-5-style rectangular cells and an orthogonal centre/portal grid.

Cells partition the inset free space exactly for rectilinear polygons. Mixed
boundary boxes are subdivided conservatively for sloping/curved boundaries;
unresolved boundary fragments are excluded at the configured resolution.
This builder selects candidate geometry only, never a route.
"""

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point

from mep_routing.geometry.polygons import iter_polygons
from .axis_grid import _edge_allowed


@dataclass(frozen=True)
class CellTemplate:
    cells: tuple
    portals: tuple
    excluded_area_mm2: float


def _merge_rectangles(rectangles, wall_bounds):
    """Merge full rectangular runs, keeping wall bands as partition features."""
    tagged = []
    for a, b, c, d in rectangles:
        x, y = (a + c) / 2, (b + d) / 2
        role = tuple(i for i, (x0, y0, x1, y1) in enumerate(wall_bounds)
                     if x0 < x < x1 and y0 < y < y1)
        tagged.append((a, b, c, d, role))
    previous_count = -1
    while len(tagged) != previous_count:
        previous_count = len(tagged)
        for vertical in (False, True):
            groups = defaultdict(list)
            for a, b, c, d, role in tagged:
                key, interval = ((a, c, role), (b, d)) if vertical else ((b, d, role), (a, c))
                groups[key].append(interval)
            merged = []
            for (first, second, role), intervals in sorted(groups.items()):
                runs = []
                for start, end in sorted(intervals):
                    if runs and abs(start - runs[-1][1]) < 1e-5:
                        runs[-1] = (runs[-1][0], end)
                    else:
                        runs.append((start, end))
                for start, end in runs:
                    merged.append((first, start, second, end, role) if vertical
                                  else (start, first, end, second, role))
            tagged = merged
    return tuple(sorted(item[:4] for item in tagged))


def _portals(cells):
    """Match opposite sides with positive-length overlap, including T junctions."""
    result = []
    for vertical in (True, False):
        starts, ends = defaultdict(list), defaultdict(list)
        for index, (x0, y0, x1, y1) in enumerate(cells):
            a, b, lo, hi = (x0, x1, y0, y1) if vertical else (y0, y1, x0, x1)
            starts[round(a, 6)].append((lo, hi, index))
            ends[round(b, 6)].append((lo, hi, index))
        for coordinate in sorted(starts.keys() & ends.keys()):
            left, right = sorted(ends[coordinate]), sorted(starts[coordinate])
            i = j = 0
            while i < len(left) and j < len(right):
                lo, hi = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
                if hi - lo > 1e-5:
                    midpoint = (lo + hi) / 2
                    point = (coordinate, midpoint) if vertical else (midpoint, coordinate)
                    result.append((left[i][2], right[j][2], point, vertical))
                if left[i][1] <= right[j][1]:
                    i += 1
                else:
                    j += 1
    return tuple(result)


def build_cell_template(allowed_region, wall_polys=(), *, boundary_resolution_mm=200):
    if not math.isfinite(boundary_resolution_mm) or boundary_resolution_mm <= 0:
        raise ValueError("cell boundary resolution must be finite and positive")
    if allowed_region is None or allowed_region.is_empty:
        return CellTemplate((), (), 0.0)
    xs, ys = set(), set()
    for polygon in iter_polygons(allowed_region.simplify(0)):
        for ring in (polygon.exterior, *polygon.interiors):
            for x, y in ring.coords:
                xs.add(round(x, 6))
                ys.add(round(y, 6))
    min_x, min_y, max_x, max_y = allowed_region.bounds
    wall_bounds = tuple(w.bounds for w in wall_polys)
    for x0, y0, x1, y1 in wall_bounds:
        xs.update(round(x, 6) for x in (x0, x1) if min_x < x < max_x)
        ys.update(round(y, 6) for y in (y0, y1) if min_y < y < max_y)
    xs, ys = sorted(xs), sorted(ys)
    if len(xs) < 2 or len(ys) < 2:
        return CellTemplate((), (), float(allowed_region.area))
    x, y = np.meshgrid(np.arange(len(xs) - 1), np.arange(len(ys) - 1))
    xs, ys = np.asarray(xs), np.asarray(ys)
    candidates = np.column_stack((xs[x.ravel()], ys[y.ravel()], xs[x.ravel() + 1], ys[y.ravel() + 1]))
    rectangles = []
    while len(candidates):
        shapes = shapely.box(*candidates.T)
        covered = shapely.covered_by(shapes, allowed_region)
        rectangles.extend(map(tuple, candidates[covered]))
        partial = candidates[~covered & shapely.intersects(shapes, allowed_region)]
        following = []
        for a, b, c, d in partial:
            if max(c - a, d - b) <= boundary_resolution_mm:
                continue
            if c - a >= d - b:
                mid = round((a + c) / 2, 6)
                following.extend(((a, b, mid, d), (mid, b, c, d)))
            else:
                mid = round((b + d) / 2, 6)
                following.extend(((a, b, c, mid), (a, mid, c, d)))
        candidates = np.asarray(following).reshape((-1, 4))
    cells = _merge_rectangles(rectangles, wall_bounds)
    area = sum((c - a) * (d - b) for a, b, c, d in cells)
    return CellTemplate(cells, _portals(cells), max(0.0, float(allowed_region.area) - area))


@lru_cache(maxsize=16)
def cached_cell_template(region_wkb, walls_wkb, boundary_resolution_mm):
    return build_cell_template(shapely.from_wkb(region_wkb), tuple(shapely.from_wkb(w) for w in walls_wkb),
                               boundary_resolution_mm=boundary_resolution_mm)


class _OrthogonalGrid:
    """Validate candidate paths, then split all crossings and overlapping spans."""

    def __init__(self, allowed, walls, wall_thickness):
        self.allowed, self.walls, self.thickness = allowed, walls, wall_thickness
        self.wall_bounds = [w.bounds for w in walls]
        self.points, self.segments = set(), set()

    def add_path(self, path):
        segments = [(tuple(a), tuple(b)) for a, b in zip(path, path[1:]) if a != b]
        if not all(_edge_allowed(LineString((a, b)), self.allowed, self.walls, self.wall_bounds, self.thickness)
                   for a, b in segments):
            return False
        self.points.update(map(tuple, path))
        self.segments.update(tuple(sorted((a, b))) for a, b in segments)
        return True

    def finish(self):
        horizontal, vertical = defaultdict(list), defaultdict(list)
        for (x0, y0), (x1, y1) in self.segments:
            if abs(y1 - y0) < 1e-5:
                horizontal[y0].append((min(x0, x1), max(x0, x1)))
            else:
                vertical[x0].append((min(y0, y1), max(y0, y1)))
        for y, intervals in horizontal.items():
            for x, spans in vertical.items():
                if any(lo <= x <= hi for lo, hi in intervals) and any(lo <= y <= hi for lo, hi in spans):
                    self.points.add((x, y))
        points = sorted(self.points)
        indices = {p: i for i, p in enumerate(points)}
        by_x, by_y = defaultdict(set), defaultdict(set)
        for x, y in points:
            by_x[x].add(y)
            by_y[y].add(x)
        edges = []
        for is_vertical, lines, points_by_line in ((False, horizontal, by_y), (True, vertical, by_x)):
            for fixed, intervals in sorted(lines.items()):
                values = sorted(points_by_line[fixed])
                for a, b in zip(values, values[1:]):
                    if b - a < 1e-5 or not any(lo <= (a + b) / 2 <= hi for lo, hi in intervals):
                        continue
                    start, end = ((fixed, a), (fixed, b)) if is_vertical else ((a, fixed), (b, fixed))
                    edges.append((indices[start], indices[end], b - a, "N" if is_vertical else "E"))
        return np.asarray(points, dtype=np.float32).reshape((-1, 2)), edges


def build_cell_skeleton(template, required_points, allowed_region, wall_polys=(), wall_thickness_mm=150,
                        *, boundary_resolution_mm=200):
    graph = _OrthogonalGrid(allowed_region, wall_polys, wall_thickness_mm)
    bounds = np.asarray(template.cells).reshape((-1, 4))
    centres = ((bounds[:, :2] + bounds[:, 2:]) / 2).tolist()
    for point in centres:
        graph.points.add(tuple(point))
    for left, right, portal, vertical in template.portals:
        for cell in (left, right):
            x, y = centres[cell]
            elbow = (x, portal[1]) if vertical else (portal[0], y)
            graph.add_path((tuple(centres[cell]), elbow, portal))
    attached = 0
    for point in sorted(set(tuple(map(float, p)) for p in required_points)):
        if not len(bounds) or not allowed_region.covers(Point(point)):
            continue
        graph.points.add(point)
        x, y = point
        distances = np.hypot(np.maximum.reduce((bounds[:, 0] - x, x - bounds[:, 2], np.zeros(len(bounds)))),
                             np.maximum.reduce((bounds[:, 1] - y, y - bounds[:, 3], np.zeros(len(bounds)))))
        containing = np.flatnonzero(distances < 1e-5)
        candidates = containing if len(containing) else np.argsort(distances, kind="stable")[:8]
        connected = False
        for cell in candidates:
            if distances[cell] > boundary_resolution_mm:
                continue
            cx, cy = centres[cell]
            # Both local L shapes are candidates; the solver decides which to use.
            connected |= graph.add_path((point, (cx, y), (cx, cy)))
            connected |= graph.add_path((point, (x, cy), (cx, cy)))
        attached += int(connected)
    nodes, edges = graph.finish()
    return nodes, edges, attached
