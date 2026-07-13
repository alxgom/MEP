from __future__ import annotations

import numpy as np


def extract_boundary_segments(region):
    """Return polygon boundary segments as an Nx4 float array."""
    segs = []

    def add_ring(coords):
        c = list(coords)
        for i in range(len(c) - 1):
            segs.append([c[i][0], c[i][1], c[i + 1][0], c[i + 1][1]])

    def add_poly(poly):
        add_ring(poly.exterior.coords)
        for interior in poly.interiors:
            add_ring(interior.coords)

    if region.geom_type == "Polygon":
        add_poly(region)
    elif region.geom_type in ("MultiPolygon", "GeometryCollection"):
        for geom in region.geoms:
            if geom.geom_type == "Polygon":
                add_poly(geom)
    return np.array(segs, dtype=np.float64) if segs else np.empty((0, 4), dtype=np.float64)


def extract_line_segments(line_geom):
    """Return line geometry segments as an Nx4 float array."""
    segs = []

    def add_coords(coords):
        c = list(coords)
        for i in range(len(c) - 1):
            segs.append([c[i][0], c[i][1], c[i + 1][0], c[i + 1][1]])

    if line_geom is None or line_geom.is_empty:
        return np.empty((0, 4), dtype=np.float64)
    if line_geom.geom_type == "LineString":
        add_coords(line_geom.coords)
    elif line_geom.geom_type == "MultiLineString" or hasattr(line_geom, "geoms"):
        for geom in line_geom.geoms:
            if geom.geom_type == "LineString":
                add_coords(geom.coords)
    return np.array(segs, dtype=np.float64) if segs else np.empty((0, 4), dtype=np.float64)

