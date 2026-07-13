from __future__ import annotations

import math

import numpy as np

from .axes import add_epsilon_axis_values, add_epsilon_geometry_axes, add_point_axes, merge_close_values


def build_axes(
    *,
    allowed_region,
    covers,
    columns,
    shafts,
    wall_polys,
    terminals,
    shaft_core_entry_specs,
    shaft_extraction,
    machine_access_points,
    epsilon_mm,
    scaffold_spacing_mm,
):
    """Build epsilon-grid axes and required connection points from active geometry."""
    xs, ys = set(), set()
    preserve_x, preserve_y = set(), set()
    required_points = []
    if allowed_region is not None:
        add_epsilon_geometry_axes(xs, ys, allowed_region, epsilon_mm)
        min_x, min_y, max_x, max_y = allowed_region.bounds
        xs.update(round(float(x)) for x in np.arange(math.floor(min_x / scaffold_spacing_mm) * scaffold_spacing_mm, math.ceil(max_x / scaffold_spacing_mm) * scaffold_spacing_mm + 1, scaffold_spacing_mm))
        ys.update(round(float(y)) for y in np.arange(math.floor(min_y / scaffold_spacing_mm) * scaffold_spacing_mm, math.ceil(max_y / scaffold_spacing_mm) * scaffold_spacing_mm + 1, scaffold_spacing_mm))
    for cover in covers:
        add_epsilon_geometry_axes(xs, ys, cover, epsilon_mm)
    for geometry in (*columns, *shafts, *wall_polys):
        add_epsilon_geometry_axes(xs, ys, geometry, epsilon_mm)

    def add_required(point):
        rounded = round(float(point[0])), round(float(point[1]))
        required_points.append(rounded)
        add_epsilon_axis_values(xs, ys, rounded, epsilon_mm)
        add_point_axes(preserve_x, preserve_y, rounded)

    for point in terminals.values():
        add_required(point)
    for spec in shaft_core_entry_specs:
        add_required(spec["entry"])
        add_required(spec["centroid"])
        if spec.get("exit_wall"):
            add_required(spec["exit_wall"][0])
            add_required(spec["exit_wall"][1])
    if shaft_extraction is not None and not shaft_core_entry_specs:
        representative = shaft_extraction.representative_point()
        add_required((representative.x, representative.y))
    for point in machine_access_points:
        add_required(point)

    return (
        merge_close_values(xs, threshold=80.0, preserve_values=preserve_x),
        merge_close_values(ys, threshold=80.0, preserve_values=preserve_y),
        required_points,
    )
