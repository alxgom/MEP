from __future__ import annotations

from vent_router.geometry import iter_polygons


def add_point_axes(xs, ys, point):
    xs.add(round(float(point[0])))
    ys.add(round(float(point[1])))


def add_polygon_vertex_axes(xs, ys, geom):
    for poly in iter_polygons(geom):
        for x, y in list(poly.exterior.coords)[:-1]:
            add_point_axes(xs, ys, (x, y))
        for interior in poly.interiors:
            for x, y in list(interior.coords)[:-1]:
                add_point_axes(xs, ys, (x, y))


def add_bounds_axes(xs, ys, geom, clearance=0.0):
    if geom is None or geom.is_empty:
        return
    buffered = geom.buffer(clearance, join_style=2) if clearance else geom
    if buffered.is_empty:
        return
    minx, miny, maxx, maxy = buffered.bounds
    xs.update([round(float(minx)), round(float(maxx))])
    ys.update([round(float(miny)), round(float(maxy))])


def merge_close_values(values, threshold, preserve_values=None, priority_values=None):
    preserve_values = {round(float(v)) for v in (preserve_values or [])}
    priority_values = {round(float(v)) for v in (priority_values or [])}
    vals = sorted({round(float(v)) for v in values})
    if not vals:
        return []

    filtered = [vals[0]]
    for i in range(1, len(vals)):
        current = vals[i]
        if current in preserve_values:
            filtered.append(current)
        elif (
            i + 1 < len(vals)
            and abs(current - vals[i + 1]) < threshold
            and (vals[i + 1] in preserve_values or (current not in priority_values and vals[i + 1] in priority_values))
        ):
            continue
        elif abs(current - filtered[-1]) >= threshold:
            filtered.append(current)

    return filtered
