"""Validated orthogonal paths with explicit crossings and overlap splitting."""

from collections import defaultdict

import numpy as np
from shapely.geometry import LineString

from .axis_grid import _edge_allowed


class OrthogonalGrid:
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
