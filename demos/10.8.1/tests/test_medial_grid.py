"""Geometry and search contracts for the experimental medial-axis grid."""

import math

import numpy as np
import pytest
from shapely.geometry import LineString, box

from mep_routing.geometry.directions import travel_direction, same_direction
from mep_routing.graphs.medial import build_medial_template, attach_medial_points, cached_medial_template
from mep_routing.graphs.runtime import create_runtime_graph
from mep_routing.graphs.dynamic import filter_dynamic_machine_obstacle
from mep_routing.routing.search import run_super_sink_state_astar, run_super_sink_line_graph_search, target_heuristic
from mep_routing.routing.metrics import count_solution_turns, merged_route_piece_lengths
from mep_routing.routing.piece_search import PieceLengthPolicy, run_piece_length_astar
from mep_routing.graphs.variants import build_medial_variant
from mep_routing.routing.clearance import add_route_interaction_weights


def reachable(adjacency, start):
    seen, todo = {start}, [start]
    while todo:
        for edge in adjacency[todo.pop()]:
            if edge[0] not in seen:
                seen.add(edge[0])
                todo.append(edge[0])
    return seen


def test_rectangle_has_a_central_axis_and_deterministic_output():
    region = box(0, 0, 4000, 2000)
    first = build_medial_template(region)
    second = build_medial_template(region)
    assert first.sample_count == 60
    np.testing.assert_array_equal(first.nodes, second.nodes)
    assert first.edges == second.edges
    assert any(abs(first.nodes[u][1] - 1000) < 1 and abs(first.nodes[v][1] - 1000) < 1
               and length >= 1500 for u, v, length, _ in first.edges)


def test_hole_keeps_a_cycle_and_no_edge_crosses_the_obstacle():
    region = box(0, 0, 5000, 4000).difference(box(1800, 1000, 3200, 3000))
    template = build_medial_template(region)
    runtime = create_runtime_graph(template.nodes, template.edges, euclidean=True)
    assert len(reachable(runtime.adjacency, 0)) == len(template.nodes)
    assert len(template.edges) >= len(template.nodes)  # Connected graph has a loop.
    for u, v, length, _ in template.edges:
        assert LineString((template.nodes[u], template.nodes[v])).covered_by(region)
        assert length == pytest.approx(math.dist(template.nodes[u], template.nodes[v]))


def test_exact_attachments_do_not_bridge_disconnected_regions():
    region = box(0, 0, 2000, 2000).union(box(3000, 0, 5000, 2000))
    template = build_medial_template(region)
    nodes, edges = attach_medial_points(template, [(200, 300), (4800, 300)], region)
    runtime = create_runtime_graph(nodes, edges, euclidean=True)
    left = next(i for i, p in enumerate(nodes) if tuple(p) == (200, 300))
    right = next(i for i, p in enumerate(nodes) if tuple(p) == (4800, 300))
    assert runtime.adjacency[left] and runtime.adjacency[right]
    assert right not in reachable(runtime.adjacency, left)


def test_template_reused_without_mutating_on_attachment():
    region = box(0, 0, 2000, 2000)
    first = cached_medial_template(region.wkb, (), 150, 200)
    before = first.nodes.copy()
    attach_medial_points(first, [(123, 456)], region)
    assert first is cached_medial_template(region.wkb, (), 150, 200)
    np.testing.assert_array_equal(first.nodes, before)
    assert not first.nodes.flags.writeable


def test_machine_hole_generates_detours_and_keeps_access_connected():
    region, machine = box(0, 0, 6000, 5000), box(2500, 3000, 3500, 4000)
    result = build_medial_variant(allowed_region=region, wall_polys=(), wall_thickness_mm=150,
                                  required_points=[(3000, 4500), (3000, 1000)], machine_polygon=machine)
    runtime = create_runtime_graph(result.nodes, result.edges, euclidean=True)
    source = int(runtime.spatial_index.query((3000, 4500))[1])
    target = int(runtime.spatial_index.query((3000, 1000))[1])
    assert target in reachable(runtime.adjacency, source)
    for u, v, _length, _direction in result.edges:
        assert LineString((result.nodes[u], result.nodes[v])).intersection(machine).length < 1e-3


def test_sampling_spacing_validation_and_empty_geometry():
    with pytest.raises(ValueError):
        build_medial_template(box(0, 0, 10, 10), sample_spacing_mm=0)
    assert not build_medial_template(box(0, 0, 0, 0)).edges


@pytest.mark.parametrize("search", [run_super_sink_state_astar, run_super_sink_line_graph_search])
def test_fixed_bend_cost_distinguishes_diagonals_with_same_cardinal_projection(search):
    # Both alternatives formerly looked like E -> E. The shorter one bends;
    # the longer weighted edge is straight. A fixed bend penalty prefers it.
    points = np.asarray([(0, 0), (10, 2), (5, 0), (5, 1)], dtype=float)
    pairs = [(0, 2), (2, 1), (0, 3), (3, 1)]
    edges = [(u, v, math.dist(points[u], points[v]), travel_direction(points[u], points[v])) for u, v in pairs]
    runtime = create_runtime_graph(points, edges, euclidean=True)
    targets = {"pin": [{"pin": "pin", "node_idx": 1, "in_dir": travel_direction(points[3], points[1]), "out_dir": "W"}]}
    weights = {(0, 3): 10.0, (1, 3): 10.0}
    path, _, _, _ = search(runtime.env, [0], ["pin"], targets, 100, edge_weights=weights)
    assert path == [0, 3, 1]
    zero = search(runtime.env, [0], ["pin"], targets, 100, edge_weights=weights, heuristic_mode=3)
    assert path == zero[0]


def test_medial_heuristic_is_euclidean_without_orthogonal_bend_estimate():
    runtime = create_runtime_graph(np.asarray([(0, 0), (3, 4)]), [], euclidean=True)
    for mode in [0, 1, 2]:
        assert target_heuristic(runtime.env, 0, "W", [{"node_idx": 1}], 100, mode, (0, 0), lambda *_: 99) == 5


def test_dynamic_machine_cannot_be_crossed_even_from_a_protected_access_node():
    points = np.asarray([(0, 0), (1000, 1000)])
    runtime = create_runtime_graph(points, [(0, 1, math.sqrt(2) * 1000, travel_direction(*points))], euclidean=True)
    env, _, blocked = filter_dynamic_machine_obstacle(runtime.nodes, runtime.edge_list, runtime.edge_coords,
                                                      box(400, 400, 600, 600), 0, {0}, euclidean=True)
    assert env.euclidean and blocked == 1
    assert not env.adj[0]


def test_diagonal_metrics_merge_straight_pieces_and_count_real_turns():
    segments = [((0, 0), (100, 100)), ((100, 100), (200, 200)), ((200, 200), (300, 200))]
    assert count_solution_turns([("Bath", segments)], include_diagonal=True) == 1
    assert merged_route_piece_lengths("Bath", segments, include_diagonal=True) == pytest.approx([math.sqrt(2) * 200, 100])
    assert same_direction((1.0, 0.0), "E")
    assert not same_direction((0.9, 0.1), "E")


def test_diagonal_candidate_uses_actual_segment_for_axis_route_interactions():
    points = np.asarray([(0, 0), (10, 10)], dtype=float)
    edge = [(0, 1, math.sqrt(200), travel_direction(*points))]
    weights = {}
    # A horizontal segment beyond the diagonal's actual crossing point must
    # not be charged merely because its box intersects the candidate's box.
    add_route_interaction_weights([((6, 5, 9, 5, "H"), 0)], 0, weights,
                                  edge, np.asarray([[0, 0, 10, 10]]), points,
                                  1.0, 100, 10, 1e9)
    assert weights == {}
    add_route_interaction_weights([((0, 5, 10, 5, "H"), 0)], 0, weights,
                                  edge, np.asarray([[0, 0, 10, 10]]), points,
                                  1.0, 100, 10, 1e9)
    assert weights[(0, 1)] == pytest.approx(math.sqrt(200) + 100)


@pytest.mark.parametrize("line_graph", [False, True])
def test_hard_piece_search_accepts_a_long_straight_diagonal(line_graph):
    points = np.asarray([(0, 0), (100, 100), (200, 200)], dtype=float)
    direction = travel_direction(points[0], points[1])
    runtime = create_runtime_graph(points, [(0, 1, math.sqrt(2)*100, direction), (1, 2, math.sqrt(2)*100, direction)], euclidean=True)
    targets = {"pin": [{"pin": "pin", "node_idx": 2, "in_dir": direction, "out_dir": "W"}]}
    result = run_piece_length_astar(runtime.env, [0], ["pin"], targets, 100,
                                    piece_policy=PieceLengthPolicy(200, 400), line_graph=line_graph)
    assert result[0] == [0, 1, 2]
