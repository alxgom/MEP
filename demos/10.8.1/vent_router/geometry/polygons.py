from __future__ import annotations


def iter_polygons(geom):
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for item in geom.geoms:
            if item.geom_type == "Polygon":
                yield item
