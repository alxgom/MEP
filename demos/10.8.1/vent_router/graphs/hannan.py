from __future__ import annotations

import math

import numpy as np

from .axes import add_bounds_axes, add_point_axes, add_polygon_vertex_axes, extend_allowed_boundary_axes


def build_static_axes(
    *,
    allowed_region,
    terminals,
    shaft_extraction,
    covers,
    columns,
    shafts,
    wall_polys,
    walls,
    grid_spacing_mm,
    scaffold_spacing_mm,
    wall_clearance_mm,
    shift_walls=False,
):
    """Build reusable Hannan axes from dwelling geometry and terminal requirements."""
    xs, ys = set(), set()
    preserve_x, preserve_y = set(), set()
    priority_x, priority_y = set(), set()

    def add_required(point):
        add_point_axes(xs, ys, point)
        add_point_axes(preserve_x, preserve_y, point)
        x, y = round(float(point[0])), round(float(point[1]))
        for delta in (-grid_spacing_mm, grid_spacing_mm):
            xs.add(x + delta)
            ys.add(y + delta)

    if allowed_region is not None:
        add_bounds_axes(xs, ys, allowed_region)
        min_x, min_y, max_x, max_y = allowed_region.bounds
        xs.update(round(float(x)) for x in np.arange(math.floor(min_x / scaffold_spacing_mm) * scaffold_spacing_mm, math.ceil(max_x / scaffold_spacing_mm) * scaffold_spacing_mm + 1, scaffold_spacing_mm))
        ys.update(round(float(y)) for y in np.arange(math.floor(min_y / scaffold_spacing_mm) * scaffold_spacing_mm, math.ceil(max_y / scaffold_spacing_mm) * scaffold_spacing_mm + 1, scaffold_spacing_mm))
        boundary_x, boundary_y = extend_allowed_boundary_axes(allowed_region)
        xs.update(boundary_x)
        ys.update(boundary_y)
        priority_x.update(boundary_x)
        priority_y.update(boundary_y)

    for point in terminals.values():
        add_required(point)
    if shaft_extraction is not None:
        representative = shaft_extraction.representative_point()
        add_required((representative.x, representative.y))
        min_x, min_y, max_x, max_y = shaft_extraction.bounds
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        for point in ((max_x + wall_clearance_mm, center_y), (min_x - wall_clearance_mm, center_y), (center_x, max_y + wall_clearance_mm), (center_x, min_y - wall_clearance_mm)):
            add_required(point)

    for cover in covers:
        inset = cover.buffer(-wall_clearance_mm, join_style=2)
        if not inset.is_empty:
            add_bounds_axes(xs, ys, inset)
            add_polygon_vertex_axes(xs, ys, inset)
    for column in columns:
        add_bounds_axes(xs, ys, column, clearance=wall_clearance_mm)
    for shaft in shafts:
        add_bounds_axes(xs, ys, shaft, clearance=wall_clearance_mm)
    for wall in wall_polys:
        add_bounds_axes(xs, ys, wall, clearance=25)

    if shift_walls:
        for wall in walls:
            coords = list(wall.coords)
            for index in range(len(coords) - 1):
                x1, y1 = float(coords[index][0]), float(coords[index][1])
                x2, y2 = float(coords[index + 1][0]), float(coords[index + 1][1])
                length = math.hypot(x2 - x1, y2 - y1)
                if length < 1.0:
                    continue
                normal_x, normal_y = -(y2 - y1) / length, (x2 - x1) / length
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                for point in ((mid_x + normal_x * wall_clearance_mm, mid_y + normal_y * wall_clearance_mm), (mid_x - normal_x * wall_clearance_mm, mid_y - normal_y * wall_clearance_mm)):
                    add_point_axes(xs, ys, point)
                    add_point_axes(priority_x, priority_y, point)

    return {
        "xs": xs,
        "ys": ys,
        "preserve_x": preserve_x,
        "preserve_y": preserve_y,
        "priority_x": priority_x,
        "priority_y": priority_y,
    }
