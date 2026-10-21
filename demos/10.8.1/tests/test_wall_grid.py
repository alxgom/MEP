"""Local wall-offset grids enrich the cell skeleton without global axes."""

import pytest
from shapely.geometry import LineString, Point, box

from mep_routing.graphs.cells import cached_cell_template
from mep_routing.graphs.runtime import create_runtime_graph
from mep_routing.graphs.variants import build_cell_variant
from mep_routing.graphs.wall_grid import build_shifted_wall_grid
from test_cell_skeleton import reachable


def test_both_wall_offsets_have_samples_and_only_cardinal_edges():
    region = box(0, 0, 4000, 2000)
    nodes, edges = build_shifted_wall_grid([LineString([(500, 1000), (3500, 1000)])], region,
                                           offset_mm=100, sample_spacing_mm=500)
    assert set(nodes[:, 1]) == {900, 1100}
    assert len(nodes) == 14
    assert all(length <= 500 and direction in ("E", "N") for u, v, length, direction in edges)
    assert (0, 900) not in set(map(tuple, nodes))  # No full-width row extension.


def test_diagonal_offsets_use_validated_orthogonal_steps():
    region = box(0, 0, 3000, 3000)
    wall = LineString([(500, 500), (2500, 2500)])
    nodes, edges = build_shifted_wall_grid([wall], region, [wall.buffer(75, cap_style=2)],
                                           offset_mm=100, sample_spacing_mm=200)
    assert len(nodes) > 10 and edges
    for u, v, _, direction in edges:
        assert direction in ("E", "N")
        assert nodes[u][0] == nodes[v][0] or nodes[u][1] == nodes[v][1]
        assert LineString((nodes[u], nodes[v])).covered_by(region)


def test_clipping_keeps_holes_and_disconnected_parts_unbridged():
    hole = box(1800, 500, 2200, 1500)
    region = box(0, 0, 4000, 2000).difference(hole)
    nodes, edges = build_shifted_wall_grid([LineString([(500, 1000), (3500, 1000)])], region,
                                           offset_mm=100, sample_spacing_mm=300)
    runtime = create_runtime_graph(nodes, edges)
    left = next(i for i, p in enumerate(nodes) if p[0] == 500)
    right = next(i for i, p in enumerate(nodes) if p[0] == 3500)
    assert right not in reachable(runtime.adjacency, left)
    assert all(LineString((nodes[u], nodes[v])).covered_by(region) for u, v, *_ in edges)


def test_wall_grid_attaches_to_an_isolated_cell_centre_and_preserves_template():
    region = box(0, 0, 4000, 2000)
    template = cached_cell_template(region.wkb, (), 200, 40000)
    before = template.cells
    result = build_cell_variant(
        allowed_region=region, wall_polys=(), wall_thickness_mm=150, required_points=[],
        walls=[LineString([(500, 1000), (3500, 1000)])], wall_offset_mm=100, wall_sample_spacing_mm=500,
    )
    runtime = create_runtime_graph(result.nodes, result.edges)
    centre = next(i for i, p in enumerate(result.nodes) if tuple(p) == (2000, 1000))
    assert len(reachable(runtime.adjacency, centre)) == len(result.nodes)
    assert result.diagnostics["wall_nodes"] == 14 and result.diagnostics["wall_links"] > 0
    assert template.cells == before


def test_wall_grid_and_cell_crossings_have_shared_nodes_and_machine_is_excluded():
    region, body = box(0, 0, 4000, 2000), box(1800, 800, 2200, 1200)
    result = build_cell_variant(
        allowed_region=region, wall_polys=(), wall_thickness_mm=150,
        required_points=[(300, 300), (3700, 1700)], machine_polygon=body,
        walls=[LineString([(500, 1000), (3500, 1000)])], wall_offset_mm=100, wall_sample_spacing_mm=500,
    )
    assert result.diagnostics["wall_nodes"]
    assert all((c - a) * (d - b) >= 40000 for a, b, c, d in result.diagnostics["cells"])
    for index, (u, v, _, direction) in enumerate(result.edges):
        assert direction in ("E", "W", "N", "S")
        line = LineString((result.nodes[u], result.nodes[v]))
        # Closed free-space boundaries permit touching; body interiors do not.
        assert line.intersection(body.buffer(-1e-3)).length < 1e-4
        for a, b, *_ in result.edges[index + 1:]:
            intersection = line.intersection(LineString((result.nodes[a], result.nodes[b])))
            if not intersection.is_empty:
                assert isinstance(intersection, Point) and {u, v} & {a, b}


def test_empty_input_and_wall_grid_parameter_validation():
    nodes, edges = build_shifted_wall_grid([], box(0, 0, 1000, 1000))
    assert nodes.shape == (0, 2) and edges == []
    with pytest.raises(ValueError):
        build_shifted_wall_grid([], box(0, 0, 1000, 1000), sample_spacing_mm=0)


def test_required_point_can_attach_to_wall_grid_when_area_floor_removes_its_cell():
    region = box(0, 0, 1000, 100)
    result = build_cell_variant(
        allowed_region=region, wall_polys=(), wall_thickness_mm=150,
        required_points=[(333, 60)], min_cell_area_mm2=200000,
        walls=[LineString([(100, 50), (900, 50)])], wall_offset_mm=20, wall_sample_spacing_mm=200,
    )
    assert not result.diagnostics["cells"] and result.diagnostics["endpoint_links"] > 0
    runtime = create_runtime_graph(result.nodes, result.edges)
    endpoint = next(i for i, p in enumerate(result.nodes) if tuple(p) == (333, 60))
    assert runtime.adjacency[endpoint]
    assert result.diagnostics["attached"] == 1
