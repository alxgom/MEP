"""Geometry assembly for terminal routing candidate regions."""

from __future__ import annotations

import numpy as np
from shapely.ops import unary_union

from mep_routing.geometry import extract_boundary_segments, extract_line_segments


def room_cover_geometry(room_polygon, covers):
    """Return the union of non-empty cover parts intersecting a room."""
    if room_polygon is None or room_polygon.is_empty or not covers:
        return None
    cover_parts = []
    for cover in covers:
        if cover is None or cover.is_empty or not cover.intersects(room_polygon):
            continue
        part = cover.intersection(room_polygon)
        if not part.is_empty and getattr(part, "area", 0.0) > 1e-6:
            cover_parts.append(part)
    return unary_union(cover_parts) if cover_parts else None


def terminal_valid_region(room_polygon, routing_region, cover_geometry):
    """Intersect a terminal room with routing and cover constraints."""
    if room_polygon is None or routing_region is None:
        return room_polygon
    valid_region = room_polygon.intersection(routing_region)
    if cover_geometry is not None and not cover_geometry.is_empty:
        valid_region = valid_region.intersection(cover_geometry)
    return valid_region


def terminal_boundary_segments(room_polygon, room_polygons, walls, wall_polygons, cover_geometry):
    """Assemble segment constraints that terminal candidates must clear."""
    segments = []

    def append_boundary(geometry):
        boundary = extract_boundary_segments(geometry)
        if len(boundary):
            segments.append(boundary)

    if room_polygon is not None:
        append_boundary(room_polygon)
    if cover_geometry is not None and not cover_geometry.is_empty:
        append_boundary(cover_geometry)
    for polygon in room_polygons:
        append_boundary(polygon)
    for wall in walls:
        line_segments = extract_line_segments(wall)
        if len(line_segments):
            segments.append(line_segments)
    for polygon in wall_polygons:
        append_boundary(polygon)
    return np.vstack(segments) if segments else np.empty((0, 4), dtype=np.float64)
