"""Assembly of Hannan and epsilon graph variants from explicit geometry inputs."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np

from .axis_grid import build_axis_grid
from .axes import merge_close_values
from .epsilon import build_axes as build_epsilon_axes


@dataclass(frozen=True)
class GraphVariantResult:
    """Uncommitted graph-builder output and useful diagnostics."""

    nodes: np.ndarray
    edges: list[tuple[int, int, float, str]]
    axes_x: list[int]
    axes_y: list[int]
    required_points: list[tuple[float, float]]
    axes_ms: float
    nodes_ms: float
    edges_ms: float


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
