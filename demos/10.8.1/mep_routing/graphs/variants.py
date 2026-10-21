"""Assembly of Hannan and epsilon graph variants from explicit geometry inputs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np
from shapely.geometry import Point

from .axis_grid import build_axis_grid
from .axes import merge_close_values
from .epsilon import build_axes as build_epsilon_axes
from .simple import SimpleGraphInputs, build_simple_grid
from .medial import cached_medial_template, attach_medial_points
from .runtime import Edge
from .cells import cached_cell_template, cached_machine_cell_template, build_cell_skeleton
from .hybrid import join_hanan_voronoi, join_grids_locally, cached_join_grids_locally
from .hannan import build_wall_shift_axes


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


def build_cell_variant(*, allowed_region, wall_polys, wall_thickness_mm, required_points,
                       boundary_resolution_mm=200, machine_polygon=None, min_cell_area_mm2=40000,
                       walls=(), wall_offset_mm=100, max_link_distance_mm=600):
    started = perf_counter()
    required_points = list(required_points)
    walls_wkb = tuple(w.wkb for w in wall_polys)
    base = cached_cell_template(allowed_region.wkb, walls_wkb, boundary_resolution_mm, min_cell_area_mm2)
    template, rebuilt_cells = base, 0
    if machine_polygon is not None:
        allowed_region = allowed_region.difference(machine_polygon)
        template, rebuilt_cells = cached_machine_cell_template(
            base, machine_polygon.wkb, walls_wkb, boundary_resolution_mm, min_cell_area_mm2, allowed_region.area,
        )
    decomposition_ms = (perf_counter() - started) * 1000
    nodes, edges, attached = build_cell_skeleton(template, required_points, allowed_region, wall_polys,
                                                wall_thickness_mm, boundary_resolution_mm=boundary_resolution_mm)
    wall_xs, wall_ys = build_wall_shift_axes(walls, wall_offset_mm)
    wall_axes_x = merge_close_values(wall_xs, threshold=120, priority_values=wall_xs)
    wall_axes_y = merge_close_values(wall_ys, threshold=120, priority_values=wall_ys)
    wall_nodes, wall_edges, _wall_timings = build_axis_grid(
        wall_axes_x, wall_axes_y, allowed_region, allowed_region, wall_polys, wall_thickness_mm,
    )
    link_info = {"links": 0, "linked_nodes": 0}
    endpoint_links = 0
    if len(wall_nodes):
        nodes, edges, link_info = join_grids_locally(
            nodes, edges, wall_nodes, wall_edges, allowed_region, wall_polys, wall_thickness_mm,
            max_link_distance_mm=max_link_distance_mm,
        )
        connected = {tuple(nodes[i]) for u, v, *_ in edges for i in (u, v)}
        # Join in both directions: an isolated cell centre/required point may
        # project onto a wall-grid edge even when no wall vertex aligns with it.
        candidates = set(map(tuple, nodes)) | set(map(tuple, required_points))
        isolated = sorted(p for p in candidates if tuple(np.asarray(p, dtype=np.float32)) not in connected
                          and allowed_region.covers(Point(p)))
        if isolated:
            nodes, edges, endpoint_info = join_grids_locally(
                nodes, edges, np.asarray(isolated), (), allowed_region, wall_polys, wall_thickness_mm,
                max_link_distance_mm=max_link_distance_mm,
            )
            endpoint_links = endpoint_info["links"]
        connected = {tuple(nodes[i]) for u, v, *_ in edges for i in (u, v)}
        attached = sum(tuple(np.asarray(p, dtype=np.float32)) in connected for p in set(map(tuple, required_points)))
    return GraphVariantResult(nodes, edges, wall_axes_x, wall_axes_y, list(required_points), decomposition_ms,
                              (perf_counter() - started) * 1000 - decomposition_ms, 0.0,
                              {"cells": template.cells, "portals": len(template.portals), "attached": attached,
                               "base_cells": len(base.cells), "rebuilt_cells": rebuilt_cells,
                               "excluded_area_mm2": template.excluded_area_mm2,
                               "discarded_small_cells": template.discarded_small_cells,
                               "min_cell_area_mm2": min_cell_area_mm2,
                               "wall_nodes": len(wall_nodes), "wall_edges": len(wall_edges),
                               "wall_links": link_info["links"],
                               "endpoint_links": endpoint_links,
                               "wall_offset_mm": wall_offset_mm,
                               "boundary_resolution_mm": boundary_resolution_mm})


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


def build_hannan_medial_variant(*, template, allowed_region, node_region, wall_polys, wall_thickness_mm,
                                terminals, shaft_extraction, machine_access_points=(),
                                sample_spacing_mm=200, machine_polygon=None, max_link_distance_mm=600):
    """Preserve both grids and connect them through local edge projections."""
    started = perf_counter()
    if machine_polygon is not None:
        allowed_region = allowed_region.difference(machine_polygon)
        node_region = node_region.difference(machine_polygon)
    medial = cached_medial_template(node_region.wkb, tuple(w.wkb for w in wall_polys),
                                    wall_thickness_mm, sample_spacing_mm)
    skeleton_ms = (perf_counter() - started) * 1000
    result = build_hannan_variant(
        template=template, allowed_region=allowed_region, node_region=node_region,
        wall_polys=wall_polys, wall_thickness_mm=wall_thickness_mm,
        terminals=terminals, shaft_extraction=shaft_extraction, machine_access_points=machine_access_points,
    )
    join_started = perf_counter()
    nodes, edges, diagnostics = join_hanan_voronoi(
        result.nodes, result.edges, medial.nodes, medial.edges, allowed_region,
        wall_polys, wall_thickness_mm, max_link_distance_mm=max_link_distance_mm,
    )
    return replace(result, nodes=nodes, edges=edges, axes_ms=result.axes_ms + skeleton_ms,
                   edges_ms=result.edges_ms + (perf_counter() - join_started) * 1000,
                   diagnostics={**diagnostics, "skeleton_nodes": len(medial.nodes),
                                "samples": medial.sample_count, "sample_spacing_mm": sample_spacing_mm,
                                "max_link_distance_mm": max_link_distance_mm})


def build_hannan_cell_variant(*, template, allowed_region, node_region, wall_polys, wall_thickness_mm,
                              terminals, shaft_extraction, required_points, machine_access_points=(),
                              machine_polygon=None, boundary_resolution_mm=200, min_cell_area_mm2=40000,
                              walls=(), wall_offset_mm=100, max_link_distance_mm=600):
    """Retain both grids' physical edges with shared junctions and local links."""
    cell_region = node_region
    if machine_polygon is not None:
        allowed_region = allowed_region.difference(machine_polygon)
        node_region = node_region.difference(machine_polygon)
    hannan = build_hannan_variant(
        template=template, allowed_region=allowed_region, node_region=node_region,
        wall_polys=wall_polys, wall_thickness_mm=wall_thickness_mm, terminals=terminals,
        shaft_extraction=shaft_extraction, machine_access_points=machine_access_points,
    )
    cells = build_cell_variant(
        allowed_region=cell_region, machine_polygon=machine_polygon,
        wall_polys=wall_polys, wall_thickness_mm=wall_thickness_mm,
        required_points=required_points, boundary_resolution_mm=boundary_resolution_mm,
        min_cell_area_mm2=min_cell_area_mm2, walls=walls, wall_offset_mm=wall_offset_mm,
        max_link_distance_mm=max_link_distance_mm,
    )
    started = perf_counter()
    nodes, edges, info = cached_join_grids_locally(
        hannan.nodes, hannan.edges, cells.nodes, cells.edges, allowed_region,
        wall_polys, wall_thickness_mm, max_link_distance_mm=max_link_distance_mm,
    )
    # Keep even isolated cell centres visible when no legal local link exists.
    present = set(map(tuple, nodes))
    missing = [p for p in cells.nodes if tuple(p) not in present]
    if missing:
        nodes = np.vstack((nodes, np.asarray(missing, dtype=np.float32)))
    return replace(
        hannan, nodes=nodes, edges=edges, required_points=cells.required_points,
        axes_ms=hannan.axes_ms + cells.axes_ms, nodes_ms=hannan.nodes_ms + cells.nodes_ms,
        edges_ms=hannan.edges_ms + cells.edges_ms + (perf_counter() - started) * 1000,
        diagnostics={**cells.diagnostics, "hannan_nodes": len(hannan.nodes), "hannan_edges": len(hannan.edges),
                     "cell_nodes": len(cells.nodes), "cell_edges": len(cells.edges), "hybrid_links": info["links"]},
    )


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
