"""Routing-core-compatible 2D polygon projection for materialized Cli ducts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

from shapely.geometry import Polygon


@dataclass(frozen=True)
class CliDuctPolygonProjection:
    conducto_polygons: tuple[Polygon, ...]
    union_polygons: tuple[Polygon, ...]
    elapsed_ms: float

    @property
    def polygons(self):
        return self.conducto_polygons + self.union_polygons


def _conducto_polygon(segment):
    start, end = segment.start, segment.end
    dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
    length = math.hypot(dx, dy)
    if length <= 0.01 or segment.width_mm <= 0:
        return None
    ox = -dy / length * segment.width_mm / 2.0
    oy = dx / length * segment.width_mm / 2.0
    return Polygon((
        (start[0] + ox, start[1] + oy), (end[0] + ox, end[1] + oy),
        (end[0] - ox, end[1] - oy), (start[0] - ox, start[1] - oy),
    ))


def _union_polygons(segments):
    widths_by_point, counts_by_point = {}, {}
    for segment in segments:
        for point in (segment.start, segment.end):
            key = (float(point[0]), float(point[1]))
            widths_by_point[key] = max(widths_by_point.get(key, 0), segment.width_mm)
            counts_by_point[key] = counts_by_point.get(key, 0) + 1
    polygons = []
    for (x, y), width in sorted(widths_by_point.items()):
        if counts_by_point[(x, y)] < 2 or width <= 0:
            continue
        half = width / 2.0
        polygons.append(Polygon((
            (x - half, y - half), (x - half, y + half),
            (x + half, y + half), (x + half, y - half),
        )))
    return tuple(polygons)


def build_core_duct_polygon_projection(segments) -> CliDuctPolygonProjection:
    """Project conducto rectangles and max-width union squares without unioning them."""
    started = perf_counter()
    segments = tuple(segments)
    conductos = tuple(
        polygon for segment in segments
        if (polygon := _conducto_polygon(segment)) is not None
    )
    return CliDuctPolygonProjection(
        conductos, _union_polygons(segments), (perf_counter() - started) * 1000.0,
    )
