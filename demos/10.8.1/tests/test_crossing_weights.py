"""Crossing costs include adapters and survive graph/route subdivision."""

import math

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from mep_routing.graphs.env import EnvView
from mep_routing.routing.clearance import add_route_interaction_weights, route_axis_records
from mep_routing.routing.piece_search import PieceLengthPolicy
from mep_routing.routing.search import run_super_sink_search


def weights_for(segments, coords, *, crossing=20, current_diameter=0, prior_diameter=0, clearance=7):
    coords = np.asarray(coords, dtype=float)
    nodes = coords.reshape(-1, 2)
    edges = [(2*i, 2*i+1, math.dist(c[:2], c[2:]), None) for i, c in enumerate(coords)]
    records = route_axis_records("Shaft", segments, lambda _name: prior_diameter, include_diagonal=True)
    weights = {}
    add_route_interaction_weights(records, current_diameter, weights, edges, coords, nodes,
                                  1, crossing, clearance, 1e6)
    return weights, edges


@pytest.mark.parametrize("segments", [
    [((0, -20), (0, 20))],
    [((0, -20), (0, -5)), ((0, -5), (0, 0)), ((0, 0), (0, 10)), ((0, 10), (0, 20))],
    [((-20, -8), (20, 12))],  # Intersection at x=-4: not an edge sample point.
])
@pytest.mark.parametrize("coords", [
    [(-10, 0, 10, 0)],
    [(-10, 0, 0, 0), (0, 0, 10, 0)],
    [(-10, 0, -4, 0), (-4, 0, 5, 0), (5, 0, 10, 0)],
])
def test_one_traversed_crossing_has_one_penalty_despite_subdivision(segments, coords):
    weights, edges = weights_for(segments, coords)
    added = sum(weights.get((a,b), length) - length for a,b,length,_ in edges)
    assert added == pytest.approx(20)
    reversed_weights, reversed_edges = weights_for(
        [(b,a) for a,b in segments], [(c[2],c[3],c[0],c[1]) for c in coords],
    )
    assert sum(reversed_weights.get((a,b),length)-length for a,b,length,_ in reversed_edges) == pytest.approx(20)


def test_diagonal_overlap_and_clearance_match_independent_geometry_oracle():
    rng = np.random.default_rng(81)
    coords = rng.integers(-50, 150, size=(200, 4)).astype(float)
    coords = np.vstack([coords, (5,5,95,35), (20,10,80,30), (30,40,90,60)])
    prior = ((5,5), (95,35))
    weights, edges = weights_for([prior], coords, current_diameter=10, prior_diameter=20)
    prior_line = LineString(prior)
    for c, (a,b,length,_) in zip(coords, edges):
        line = LineString([c[:2], c[2:]])
        intersection = line.intersection(prior_line)
        if intersection.length > 1e-7:
            expected = 1e6
        elif not intersection.is_empty:
            at_endpoint = min(Point(c[:2]).distance(intersection), Point(c[2:]).distance(intersection)) < 1e-7
            expected = length + (10 if at_endpoint else 20)
        elif line.distance(prior_line) < 15:
            expected = length + 7
        else:
            expected = length
        assert weights.get((a,b),length) == pytest.approx(expected)


@pytest.mark.parametrize("backend", [0,1,3,4])
@pytest.mark.parametrize("crossing", [0,100,300,340,1000])
def test_normal_and_length_aware_astar_switch_at_the_same_crossing_cost(backend, crossing):
    nodes = np.array([(-100,0), (0,0), (100,0), (-100,100), (100,100)], dtype=float)
    pairs = [(0,1),(1,2),(0,3),(3,4),(4,2)]
    adjacency = {i:[] for i in range(len(nodes))}
    edges = []
    for a,b in pairs:
        dx,dy = nodes[b] - nodes[a]
        direction = ("E" if dx>0 else "W") if dy==0 else ("N" if dy>0 else "S")
        distance = math.dist(nodes[a], nodes[b])
        edges.append((a,b,distance,direction))
        adjacency[a].append((b,distance,direction))
        adjacency[b].append((a,distance,{"E":"W","W":"E","N":"S","S":"N"}[direction]))
    coords = np.array([(*nodes[a],*nodes[b]) for a,b in pairs])
    records = route_axis_records("Shaft", [((0,-20),(0,0)),((0,0),(0,20))], lambda _name:0, include_diagonal=True)
    weights = {}
    add_route_interaction_weights(records,0,weights,edges,coords,nodes,1,crossing,0,1e6)
    original = dict(weights)
    target = {"pin":"p","node_idx":2,"in_dir":"E","out_dir":"W"}
    result = run_super_sink_search(backend,EnvView(nodes,adjacency),[0],["p"],{"p":[target]},40,
                                  edge_weights=weights,heuristic_mode=3,piece_policy=PieceLengthPolicy(25,50))
    assert result[0] == ([0,1,2] if crossing < 320 else [0,3,4,2])
    assert weights == original
