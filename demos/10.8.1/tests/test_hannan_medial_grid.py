"""Sparse Hanan/Voronoi joining and routing contracts."""

import math

import numpy as np
import pytest
from shapely.geometry import LineString, box

from mep_routing.geometry.directions import travel_direction
from mep_routing.graphs.hybrid import join_hanan_voronoi
from mep_routing.graphs.lifecycle import GraphMode
from mep_routing.graphs.runtime import create_runtime_graph
from mep_routing.graphs.variants import build_hannan_variant, build_hannan_medial_variant
from mep_routing.routing.search import run_super_sink_state_astar, run_super_sink_line_graph_search
from test_graph_lifecycle import make_lifecycle


def records(points, pairs):
    return [(u, v, math.dist(points[u], points[v]), travel_direction(points[u], points[v])) for u, v in pairs]


def reachable(adjacency, start):
    seen, todo = {start}, [start]
    while todo:
        for neighbour, *_ in adjacency[todo.pop()]:
            if neighbour not in seen:
                seen.add(neighbour)
                todo.append(neighbour)
    return seen


@pytest.mark.parametrize("search", [run_super_sink_state_astar, run_super_sink_line_graph_search])
def test_local_links_split_hanan_edges_and_enable_switching_to_diagonal(search):
    base = np.asarray([(0, 0), (1000, 0), (1000, 1000), (0, 1000)], dtype=np.float32)
    medial = np.asarray([(300, 300), (700, 700)], dtype=np.float32)
    nodes, edges, info = join_hanan_voronoi(
        base, records(base, [(0, 1), (1, 2), (2, 3), (3, 0)]), medial, records(medial, [(0, 1)]),
        box(-10, -10, 1010, 1010), max_link_distance_mm=350,
    )
    index = {tuple(p): i for i, p in enumerate(nodes)}
    assert info == {"links": 4, "linked_nodes": 2}
    # Local segments stop at the first edge: no Cartesian row/column expansion.
    assert {(300, 0), (0, 300), (700, 1000), (1000, 700)} <= set(index)
    assert not {(300, 1000), (1000, 300), (700, 0), (0, 700)} & set(index)
    runtime = create_runtime_graph(nodes, edges, euclidean=True)
    assert len(runtime.adjacency[index[(300, 0)]]) == 3
    assert index[(700, 700)] in reachable(runtime.adjacency, index[(0, 0)])
    direction = travel_direction(medial[0], medial[1])
    targets = {"pin": [{"pin": "pin", "node_idx": index[(700, 700)], "in_dir": direction, "out_dir": "W"}]}
    path, *_ = search(runtime.env, [index[(300, 0)]], ["pin"], targets, 10)
    assert index[(300, 300)] in path
    assert path[-1] == index[(700, 700)]
    assert any(isinstance(d, tuple) for *_, d in edges)


def test_natural_diagonal_crossing_is_a_shared_node_even_without_nearby_links():
    base = np.asarray([(500, -1000), (500, 1000)])
    medial = np.asarray([(100, -400), (900, 400)])
    nodes, edges, info = join_hanan_voronoi(base, records(base, [(0, 1)]), medial,
                                           records(medial, [(0, 1)]), box(0, -1100, 1000, 1100),
                                           max_link_distance_mm=1)
    runtime = create_runtime_graph(nodes, edges, euclidean=True)
    index = {tuple(p): i for i, p in enumerate(nodes)}
    assert info["links"] == 0
    assert len(runtime.adjacency[index[(500, 0)]]) == 4
    assert len(reachable(runtime.adjacency, 0)) == len(nodes)


def test_collinear_overlap_is_split_and_deduplicated():
    base, medial = np.asarray([(0, 0), (1000, 0)]), np.asarray([(250, 0), (750, 0)])
    nodes, edges, _ = join_hanan_voronoi(base, records(base, [(0, 1)]), medial,
                                         records(medial, [(0, 1)]), box(-1, -1, 1001, 1))
    assert len(nodes) == 4 and len(edges) == 3
    assert sum(edge[2] for edge in edges) == 1000


def test_local_links_cannot_bridge_holes_components_or_long_wall_runs():
    base = np.asarray([(500, 100), (500, 900)])
    medial = np.asarray([(1500, 300), (1500, 700)])
    base_edges, medial_edges = records(base, [(0, 1)]), records(medial, [(0, 1)])
    regions = [box(0, 0, 2000, 1000).difference(box(800, 0, 1200, 1000)), box(0, 0, 2000, 1000)]
    for region, walls in [(regions[0], ()), (regions[1], (box(800, 0, 1200, 1000),))]:
        nodes, edges, info = join_hanan_voronoi(base, base_edges, medial, medial_edges, region,
                                               walls, 150, max_link_distance_mm=1200)
        assert info["links"] == 0
        runtime = create_runtime_graph(nodes, edges, euclidean=True)
        source = next(i for i, p in enumerate(nodes) if tuple(p) == (1500, 300))
        assert 0 not in reachable(runtime.adjacency, source)
        assert all(LineString((nodes[u], nodes[v])).covered_by(region) for u, v, *_ in edges)


def test_hybrid_preserves_hanan_axes_machine_hole_and_cached_voronoi_geometry():
    region, machine = box(0, 0, 4000, 3000), box(1600, 1000, 2400, 2000)
    template = {"xs": {100, 500, 900, 3500}, "ys": {100, 500, 900, 2500},
                "preserve_x": {100, 500, 900, 3500}, "preserve_y": {100, 500, 900, 2500},
                "priority_x": set(), "priority_y": set()}
    before = {key: values.copy() for key, values in template.items()}
    kwargs = dict(template=template, allowed_region=region, node_region=region, wall_polys=(),
                  wall_thickness_mm=150, terminals={"t": (500, 500)}, shaft_extraction=None)
    baseline = build_hannan_variant(**kwargs)
    first = build_hannan_medial_variant(**kwargs, machine_polygon=machine)
    again = build_hannan_medial_variant(**kwargs, machine_polygon=machine)
    assert template == before
    assert first.axes_x == baseline.axes_x and first.axes_y == baseline.axes_y
    np.testing.assert_array_equal(first.nodes, again.nodes)
    assert first.edges == again.edges
    assert any(isinstance(direction, tuple) for *_, direction in first.edges)
    assert all(LineString((first.nodes[u], first.nodes[v])).intersection(machine).length < 1e-3
               for u, v, *_ in first.edges)


def test_hybrid_lifecycle_enables_euclidean_heuristics_and_exact_access():
    lifecycle = make_lifecycle()
    result = lifecycle.build_selected(GraphMode.HANNAN_MEDIAL, {"supply": (300, 500)}, 0)
    assert int(GraphMode.HANNAN_MEDIAL) == 6
    assert result.runtime.env.euclidean and result.runtime.env.exact_access
    assert (300, 500) in set(map(tuple, result.runtime.nodes))
    assert result.variant.diagnostics["links"] > 0


def test_empty_grids_and_invalid_link_radius():
    empty = np.empty((0, 2))
    nodes, edges, _ = join_hanan_voronoi(empty, [], empty, [], box(0, 0, 1, 1))
    assert nodes.shape == (0, 2) and edges == []
    with pytest.raises(ValueError):
        join_hanan_voronoi(empty, [], empty, [], box(0, 0, 1, 1), max_link_distance_mm=0)
