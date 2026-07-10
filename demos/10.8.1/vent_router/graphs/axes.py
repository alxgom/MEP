from __future__ import annotations

import numpy as np
from shapely.geometry import Point

from vent_router.geometry import iter_polygons, largest_polygon


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


def add_epsilon_axis_values(xs, ys, point, epsilon):
    x = round(float(point[0]))
    y = round(float(point[1]))
    for dx in (-epsilon, 0.0, epsilon):
        xs.add(round(x + dx))
    for dy in (-epsilon, 0.0, epsilon):
        ys.add(round(y + dy))


def add_epsilon_geometry_axes(xs, ys, geom, epsilon):
    if geom is None or geom.is_empty:
        return
    for poly in iter_polygons(geom):
        for x, y in list(poly.exterior.coords)[:-1]:
            add_epsilon_axis_values(xs, ys, (x, y), epsilon)
        for interior in poly.interiors:
            for x, y in list(interior.coords)[:-1]:
                add_epsilon_axis_values(xs, ys, (x, y), epsilon)
        minx, miny, maxx, maxy = poly.bounds
        for pt in ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)):
            add_epsilon_axis_values(xs, ys, pt, epsilon)


def extend_allowed_boundary_axes(allowed, inset=100.0, cluster_dist=300.0):
    poly = largest_polygon(allowed)
    if poly is None:
        return [], []

    pts = list(poly.exterior.coords)[:-1]
    if len(pts) < 3:
        return [], []

    sharp_points = []
    for i, p2 in enumerate(pts):
        p1 = np.array(pts[i - 1], dtype=np.float64)
        p2_arr = np.array(p2, dtype=np.float64)
        p3 = np.array(pts[(i + 1) % len(pts)], dtype=np.float64)
        v1 = p1 - p2_arr
        v2 = p3 - p2_arr
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_a = np.dot(v1, v2) / (n1 * n2)
        angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        if angle < 170.0:
            sharp_points.append((p2_arr, v1 / n1, v2 / n2))

    interior = []
    inset_poly = poly.buffer(-10.0)
    for p, v1, v2 in sharp_points:
        direction = v1 + v2
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            moved = p
        else:
            direction = direction / norm
            cand = p + direction * inset
            if poly.contains(Point(float(cand[0]), float(cand[1]))):
                moved = cand
            else:
                cand = p - direction * inset
                moved = cand if poly.contains(Point(float(cand[0]), float(cand[1]))) else p
        if inset_poly.contains(Point(float(moved[0]), float(moved[1]))):
            interior.append(moved)

    if not interior:
        return [], []

    coords = np.array(interior, dtype=np.float64)
    used = np.zeros(len(coords), dtype=bool)
    clusters = []
    for i in range(len(coords)):
        if used[i]:
            continue
        dist = np.linalg.norm(coords - coords[i], axis=1)
        idxs = np.where(dist < cluster_dist)[0]
        used[idxs] = True
        clusters.append(coords[idxs].mean(axis=0))

    xs = sorted({round(float(p[0])) for p in clusters})
    ys = sorted({round(float(p[1])) for p in clusters})
    return xs, ys


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
