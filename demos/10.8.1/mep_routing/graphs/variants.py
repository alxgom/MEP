"""Assembly of Hannan and epsilon graph variants from explicit geometry inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np

from .axis_grid import build_axis_grid
from .axes import merge_close_values
from .epsilon import build_axes as build_epsilon_axes
from .simple import SimpleGraphInputs, build_simple_grid
from .medial import cached_medial_template, attach_medial_points
from .runtime import Edge


@dataclass(frozen=True)
class GraphVariantResult:
    """Uncommitted graph-builder output and useful diagnostics."""

    nodes: np.ndarray
    edges: list[Edge]
    axes_x: list[int]
    axes_y: list[int]
    required_points: list[tuple[float, float]]
    axes_ms: float
    nodes_ms: float
    edges_ms: float
    diagnostics: dict = field(default_factory=dict)


def build_medial_variant(*, allowed_region, wall_polys, wall_thickness_mm,
                         required_points, sample_spacing_mm=200, machine_polygon=None):
    started = perf_counter()
    if machine_polygon is not None:
        # Cutting a cached centreline alone can strand a port. Include the body
        # as a polygon hole so this pose gets candidate detours around it.
        allowed_region = allowed_region.difference(machine_polygon)
    template = cached_medial_template(allowed_region.wkb, tuple(w.wkb for w in wall_polys),
                                     wall_thickness_mm, sample_spacing_mm)
    skeleton_ms = (perf_counter() - started) * 1000
    nodes, edges = attach_medial_points(template, required_points, allowed_region, wall_polys,
                                       wall_thickness_mm, machine_polygon=machine_polygon)
    attach_ms = (perf_counter() - started) * 1000 - skeleton_ms
    return GraphVariantResult(nodes, edges, [], [], list(required_points), skeleton_ms, attach_ms, 0.0,
                              {"samples": template.sample_count, "skeleton_nodes": len(template.nodes),
                               "sample_spacing_mm": sample_spacing_mm})


def build_hannan_variant(
    *,
    template: Mapping[str, Iterable[float]],
    allowed_region,
    node_region,
    wall_polys,
    wall_thickness_mm: float,
    terminals: Mapping[str, tuple[float, float]],
    shaft_extraction,
    machine_access_points: Iterable[tuple[float, float]] = (),
    merge_threshold_mm: float = 120.0,
) -> GraphVariantResult:
    """Build a Hannan graph from a cached static-axis template and live ports."""
    started = perf_counter()
    xs = set(template["xs"])
    ys = set(template["ys"])
    preserve_x = set(template["preserve_x"])
    preserve_y = set(template["preserve_y"])
    required_points = [(round(float(point[0])), round(float(point[1]))) for point in terminals.values()]
    if shaft_extraction is not None:
        representative = shaft_extraction.representative_point()
        required_points.append((round(float(representative.x)), round(float(representative.y))))

    for point in machine_access_points:
        x, y = point
        xs.add(x)
        ys.add(y)
        preserve_x.add(x)
        preserve_y.add(y)
        required_points.append((x, y))

    axes_x = merge_close_values(
        xs,
        threshold=merge_threshold_mm,
        preserve_values=preserve_x,
        priority_values=template["priority_x"],
    )
    axes_y = merge_close_values(
        ys,
        threshold=merge_threshold_mm,
        preserve_values=preserve_y,
        priority_values=template["priority_y"],
    )
    axes_ms = (perf_counter() - started) * 1000.0
    nodes, edges, (nodes_ms, edges_ms) = build_axis_grid(
        axes_x,
        axes_y,
        allowed_region,
        node_region,
        wall_polys,
        wall_thickness_mm,
        required_points,
    )
    return GraphVariantResult(nodes, edges, axes_x, axes_y, required_points, axes_ms, nodes_ms, edges_ms)


def build_epsilon_variant(
    *,
    allowed_region,
    node_region,
    covers,
    columns,
    shafts,
    wall_polys,
    wall_thickness_mm: float,
    terminals: Mapping[str, tuple[float, float]],
    shaft_core_entry_specs: Iterable[dict[str, Any]],
    shaft_extraction,
    machine_access_points: Iterable[tuple[float, float]] = (),
    epsilon_mm: float,
    scaffold_spacing_mm: float,
) -> GraphVariantResult:
    """Build an epsilon graph from active geometry and live machine access ports."""
    started = perf_counter()
    axes_x, axes_y, required_points = build_epsilon_axes(
        allowed_region=allowed_region,
        covers=covers,
        columns=columns,
        shafts=shafts,
        wall_polys=wall_polys,
        terminals=terminals,
        shaft_core_entry_specs=shaft_core_entry_specs,
        shaft_extraction=shaft_extraction,
        machine_access_points=machine_access_points,
        epsilon_mm=epsilon_mm,
        scaffold_spacing_mm=scaffold_spacing_mm,
    )
    axes_ms = (perf_counter() - started) * 1000.0
    nodes, edges, (nodes_ms, edges_ms) = build_axis_grid(
        axes_x,
        axes_y,
        allowed_region,
        node_region,
        wall_polys,
        wall_thickness_mm,
        required_points,
    )
    return GraphVariantResult(nodes, edges, axes_x, axes_y, required_points, axes_ms, nodes_ms, edges_ms)


def build_simple_variant(inputs: SimpleGraphInputs) -> GraphVariantResult:
    result = build_simple_grid(inputs)
    return GraphVariantResult(
        result.nodes, result.edges, result.axes_x, result.axes_y,
        result.required_points, result.axes_ms, result.nodes_ms, result.edges_ms,
    )
