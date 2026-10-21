"""A physical union of Hanan and cell grids, separate from search strategy."""

import numpy as np
import pytest
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

from mep_routing.graphs.lifecycle import GraphMode
from mep_routing.graphs.hybrid import cached_join_grids_locally
from mep_routing.graphs.runtime import create_runtime_graph
from mep_routing.graphs.variants import build_cell_variant, build_hannan_variant, build_hannan_cell_variant
from mep_routing.routing.search import run_super_sink_state_astar, run_super_sink_line_graph_search
from test_cell_skeleton import reachable
from test_graph_lifecycle import make_lifecycle


def inputs():
    region = box(0, 0, 4000, 3000)
    xs, ys = {300, 500, 1500, 2500, 3500, 3700}, {300, 500, 1500, 2500, 2700}
    return dict(template={"xs": xs, "ys": ys, "preserve_x": xs.copy(), "preserve_y": ys.copy(),
                          "priority_x": set(), "priority_y": set()},
                allowed_region=region, node_region=region, wall_polys=(), wall_thickness_mm=150,
                terminals={"t": (300, 300)}, shaft_extraction=None, required_points=[(700, 900), (3700, 2700)])


def source_grids(kwargs):
    hannan = build_hannan_variant(**{key: kwargs[key] for key in (
        "template", "allowed_region", "node_region", "wall_polys", "wall_thickness_mm", "terminals", "shaft_extraction")})
    cells = build_cell_variant(allowed_region=kwargs["node_region"], wall_polys=kwargs["wall_polys"],
                               wall_thickness_mm=kwargs["wall_thickness_mm"], required_points=kwargs["required_points"])
    return hannan, cells


def test_both_edge_sets_survive_as_split_geometry_without_expanding_hanan_axes():
    kwargs = inputs()
    hannan, cells = source_grids(kwargs)
    result = build_hannan_cell_variant(**kwargs)
    geometry = unary_union([LineString((result.nodes[u], result.nodes[v])) for u, v, *_ in result.edges])
    for source in (hannan, cells):
        assert set(map(tuple, source.nodes)) <= set(map(tuple, result.nodes))
        for u, v, *_ in source.edges:
            assert LineString((source.nodes[u], source.nodes[v])).difference(geometry).length < 1e-4
    assert result.axes_x == hannan.axes_x and result.axes_y == hannan.axes_y
    assert 2000 not in result.axes_x  # The cell-centre X was not made a global axis.
    assert result.diagnostics["cells"] == cells.diagnostics["cells"]
    assert result.diagnostics["min_cell_area_mm2"] == 40000
    assert result.diagnostics["hybrid_links"] > 0
    for index, (u, v, _, direction) in enumerate(result.edges):
        assert direction in ("E", "W", "N", "S")
        line = LineString((result.nodes[u], result.nodes[v]))
        for a, b, *_ in result.edges[index + 1:]:
            intersection = line.intersection(LineString((result.nodes[a], result.nodes[b])))
            if not intersection.is_empty:
                assert isinstance(intersection, Point) and {u, v} & {a, b}


@pytest.mark.parametrize("search", [run_super_sink_state_astar, run_super_sink_line_graph_search])
def test_both_search_representations_route_between_hanan_and_cell_points(search):
    result = build_hannan_cell_variant(**inputs())
    runtime = create_runtime_graph(result.nodes, result.edges, exact_access=True)
    indices = {tuple(p): i for i, p in enumerate(result.nodes)}
    start, target = indices[(500, 500)], indices[(700, 900)]
    targets = {"pin": [{"pin": "pin", "node_idx": target, "in_dir": "E", "out_dir": "W"}]}
    path, *_ = search(runtime.env, [start], ["pin"], targets, 10)
    assert path[0] == start and path[-1] == target
    assert all(any(edge[0] == v for edge in runtime.adjacency[u]) for u, v in zip(path, path[1:]))
    assert not runtime.env.euclidean


def test_machine_and_disconnected_regions_are_respected_and_sources_are_immutable():
    kwargs = inputs()
    region = kwargs["allowed_region"].union(box(5000, 0, 7000, 3000))
    kwargs.update(allowed_region=region, node_region=region,
                  required_points=[(700, 900), (6000, 900)], machine_polygon=box(1800, 1000, 2200, 2000))
    kwargs["template"]["xs"].update((5500, 6500))
    before = {key: value.copy() for key, value in kwargs["template"].items()}
    first = build_hannan_cell_variant(**kwargs)
    again = build_hannan_cell_variant(**kwargs)
    assert kwargs["template"] == before
    np.testing.assert_array_equal(first.nodes, again.nodes)
    assert first.edges == again.edges
    runtime = create_runtime_graph(first.nodes, first.edges)
    indices = {tuple(p): i for i, p in enumerate(first.nodes)}
    assert indices[(6000, 900)] not in reachable(runtime.adjacency, indices[(700, 900)])
    for u, v, *_ in first.edges:
        line = LineString((first.nodes[u], first.nodes[v]))
        assert line.covered_by(region)
        assert line.intersection(kwargs["machine_polygon"].buffer(-1e-3)).length < 1e-4


def test_lifecycle_keeps_orthogonal_costs_exact_ports_and_cell_overlay():
    lifecycle = make_lifecycle()
    result = lifecycle.build_selected(GraphMode.HANNAN_CELL, {"supply": (300, 500)}, 0)
    assert int(GraphMode.HANNAN_CELL) == 7
    assert not result.runtime.env.euclidean and result.runtime.env.exact_access
    assert result.runtime.env.cells
    assert (300, 500) in set(map(tuple, result.runtime.nodes))
    filtered = lifecycle.apply_dynamic_obstacle(result.runtime, box(800, 800, 900, 900), {}, 0, clearance_mm=0)
    assert filtered.env.cells == result.runtime.env.cells
    assert not filtered.env.euclidean and filtered.env.exact_access


@pytest.mark.parametrize("empty_source", ["hanan", "cells"])
def test_a_missing_input_layer_does_not_remove_the_other_grid(empty_source):
    kwargs = inputs()
    if empty_source == "hanan":
        kwargs["template"] = {key: set() for key in kwargs["template"]}
    else:
        kwargs["min_cell_area_mm2"] = 100000000
    result = build_hannan_cell_variant(**kwargs)
    assert result.edges
    assert result.diagnostics["hannan_nodes" if empty_source == "hanan" else "cell_nodes"] == 0
    assert result.diagnostics["cell_nodes" if empty_source == "hanan" else "hannan_nodes"] > 0


def test_union_cache_is_immutable_and_keys_edge_geometry_walls_and_radius():
    base = np.asarray([(0, 0), (0, 1000)], dtype=np.float32)
    added = np.asarray([(300, 500)], dtype=np.float32)
    edges = [(0, 1, 1000.0, "N")]
    region = box(-10, -10, 1010, 1010)
    nodes, records, info = cached_join_grids_locally(base, edges, added, [], region)
    expected_records = records.copy()
    records.clear()
    info["links"] = -1
    again, records, info = cached_join_grids_locally(base, edges, added, [], region)
    assert again is nodes and not nodes.flags.writeable
    assert records == expected_records and info["links"] == 1
    with pytest.raises(ValueError):
        nodes[0, 0] = 999
    short = cached_join_grids_locally(base, edges, added, [], region, max_link_distance_mm=200)
    assert short[2]["links"] == 0
    wall = box(100, 0, 200, 1000)
    blocked = cached_join_grids_locally(base, edges, added, [], region, [wall], 20)
    assert blocked[2]["links"] == 0
    relaxed = cached_join_grids_locally(base, edges, added, [], region, [wall], 150)
    assert relaxed[2]["links"] == 1
    clipped = cached_join_grids_locally(base, edges, added, [], region.difference(wall))
    assert clipped[2]["links"] == 0
    moved = cached_join_grids_locally(base + (100, 0), edges, added, [], region)
    assert tuple(moved[0][0]) == (100, 0)
    altered = cached_join_grids_locally(base, [], added, [], region)
    assert not altered[1]
