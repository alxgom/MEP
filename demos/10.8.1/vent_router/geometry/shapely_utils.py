from __future__ import annotations

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


def snap_to_integer_grid(geom):
    """Round supported Shapely geometry coordinates to integer millimetres."""
    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        ext = [(round(x), round(y)) for x, y in geom.exterior.coords]
        ints = [
            [(round(x), round(y)) for x, y in interior.coords]
            for interior in geom.interiors
        ]
        return Polygon(ext, ints)
    if geom.geom_type == "LineString":
        return LineString([(round(x), round(y)) for x, y in geom.coords])
    if geom.geom_type in ("MultiLineString", "MultiPolygon", "GeometryCollection"):
        return unary_union([snap_to_integer_grid(g) for g in geom.geoms])
    return geom

