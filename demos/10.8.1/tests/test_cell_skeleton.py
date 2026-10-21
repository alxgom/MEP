"""Contracts for the cell-based orthogonal skeleton illustrated by Figure 5."""

import math

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from mep_routing.graphs.cells import build_cell_template, build_cell_skeleton, cached_cell_template, _portals
from mep_routing.graphs.lifecycle import GraphLifecycle, GraphMode
from mep_routing.graphs.runtime import create_runtime_graph
from mep_routing.graphs.variants import build_cell_variant
from mep_routing.routing.search import target_heuristic


def reachable(adjacency, source):
    seen, todo = {source}, [source]
    while todo:
        for other, _length, _direction in adjacency[todo.pop()]:
            if other not in seen:
                seen.add(other)
                todo.append(other)
    return seen


def test_empty_room_has_one_cell_and_a_centre_node():
    region = box(0, 0, 4000, 2000)
    template = build_cell_template(region)
    nodes, edges, attached = build_cell_skeleton(template, [(100, 400), (3900, 1600)], region)
    assert template.cells == ((0, 0, 4000, 2000),)
    assert not template.portals and not template.excluded_area_mm2
    assert (2000, 1000) in set(map(tuple, nodes))
    assert attached == 2
    assert edges


def test_rectilinear_decomposition_covers_free_space_without_overlapping_cells():
    region = box(0, 0, 5000, 4000).difference(box(1800, 1000, 3200, 3000))
    template = build_cell_template(region)
    cells = [box(*bounds) for bounds in template.cells]
    assert unary_union(cells).symmetric_difference(region).area < 1e-5
    assert sum(c.area for c in cells) == pytest.approx(region.area)
    assert len(template.portals) >= len(template.cells)  # Retain both ways round the hole.
    nodes, edges, _ = build_cell_skeleton(template, [], region)
    runtime = create_runtime_graph(nodes, edges)
    assert len(reachable(runtime.adjacency, 0)) == len(nodes)
    for u, v, length, direction in edges:
        assert LineString((nodes[u], nodes[v])).covered_by(region)
        assert nodes[u][0] == nodes[v][0] or nodes[u][1] == nodes[v][1]
        assert length == math.dist(nodes[u], nodes[v])
        assert direction in ("E", "N")


def test_sloping_boundary_is_conservative_and_refinement_reduces_excluded_area():
    region = Polygon([(0, 0), (5000, 0), (4000, 4000), (0, 4000)])
    coarse = build_cell_template(region, boundary_resolution_mm=400, min_cell_area_mm2=0)
    fine = build_cell_template(region, boundary_resolution_mm=100, min_cell_area_mm2=0)
    assert 0 < fine.excluded_area_mm2 < coarse.excluded_area_mm2
    assert all(box(*bounds).covered_by(region) for bounds in fine.cells)
    assert region.area - sum(box(*b).area for b in fine.cells) == pytest.approx(fine.excluded_area_mm2)


def test_unequal_cells_share_a_portal_but_corner_contact_does_not_connect():
    cells = ((0, 0, 1000, 2000), (1000, 0, 2000, 500), (1000, 500, 2000, 2000), (2000, 2000, 3000, 3000))
    portals = _portals(cells)
    assert len(portals) == 3
    assert any({a, b} == {0, 1} and point == (1000, 250) for a, b, point, _ in portals)
    assert not any(a == 3 or b == 3 for a, b, _point, _axis in portals)


def test_wall_bands_remain_cell_features_without_becoming_solid_obstacles():
    region, wall = box(0, 0, 4000, 2000), box(1900, 0, 2100, 2000)
    template = build_cell_template(region, [wall])
    assert len(template.cells) == 3
    nodes, edges, attached = build_cell_skeleton(template, [(500, 1000), (3500, 1000)], region, [wall], 200)
    runtime = create_runtime_graph(nodes, edges)
    left, right = [int(runtime.spatial_index.query(p)[1]) for p in [(500, 1000), (3500, 1000)]]
    assert attached == 2 and right in reachable(runtime.adjacency, left)


def test_required_attachments_are_exact_and_do_not_bridge_disconnected_regions():
    region = box(0, 0, 2000, 2000).union(box(3000, 0, 5000, 2000))
    template = build_cell_template(region)
    nodes, edges, attached = build_cell_skeleton(template, [(123, 456), (4321, 765)], region)
    runtime = create_runtime_graph(nodes, edges)
    left, right = [int(runtime.spatial_index.query(p)[1]) for p in [(123, 456), (4321, 765)]]
    assert attached == 2
    assert tuple(nodes[left]) == (123, 456) and tuple(nodes[right]) == (4321, 765)
    assert right not in reachable(runtime.adjacency, left)


def test_all_crossings_and_t_junctions_are_explicit_nodes_without_overlapping_edges():
    region = box(0, 0, 4000, 2000)
    template = build_cell_template(region)
    nodes, edges, _ = build_cell_skeleton(template, [(100, 300), (3500, 300), (3500, 1700)], region)
    for index, (u, v, _length, _direction) in enumerate(edges):
        for a, b, _l, _d in edges[index + 1:]:
            intersection = LineString((nodes[u], nodes[v])).intersection(LineString((nodes[a], nodes[b])))
            if intersection.is_empty:
                continue
            assert isinstance(intersection, Point)
            assert {u, v} & {a, b}


def test_machine_body_is_a_hole_and_cached_cells_do_not_change_with_attachments():
    region, body = box(0, 0, 6000, 5000), box(2500, 3000, 3500, 4000)
    template = cached_cell_template(region.difference(body).wkb, (), 200)
    before = template.cells
    result = build_cell_variant(allowed_region=region, wall_polys=(), wall_thickness_mm=150,
                                required_points=[(3000, 4500), (3000, 1000)], machine_polygon=body)
    runtime = create_runtime_graph(result.nodes, result.edges)
    a, b = [int(runtime.spatial_index.query(p)[1]) for p in [(3000, 4500), (3000, 1000)]]
    assert b in reachable(runtime.adjacency, a)
    assert template.cells == before
    assert template is cached_cell_template(region.difference(body).wkb, (), 200)
    assert all(LineString((result.nodes[u], result.nodes[v])).intersection(body).length < 1e-5
               for u, v, _length, _direction in result.edges)


def test_mode_keeps_manhattan_heuristics_and_preserves_cell_overlay_after_filtering():
    lifecycle = GraphLifecycle(box(0, 0, 4000, 2000), (), (), (), (), (), {"Bath": (500, 500)}, None, (),
                               200, 600, 150, 0, 200,
                               lambda pins, angle: [{"pin": "supply", "out_dir": "E", "access_point": (3500, 1500)}])
    result = lifecycle.build_selected(GraphMode.CELL_SKELETON, {}, 0)
    assert int(GraphMode.MEDIAL_AXIS) == 4 and int(GraphMode.CELL_SKELETON) == 5
    assert not result.runtime.env.euclidean and result.runtime.env.exact_access
    assert result.runtime.env.cells
    env = lifecycle.apply_dynamic_obstacle(result.runtime, box(3900, 1900, 3950, 1950), {}, 0, clearance_mm=0).env
    assert env.cells == result.runtime.env.cells and env.exact_access
    synthetic = create_runtime_graph(np.asarray([(0, 0), (3, 4)]), [])
    assert target_heuristic(synthetic.env, 0, "E", [{"node_idx": 1}], 100, 1, (0, 0), lambda *_: 0) == 7


def test_empty_domain_and_resolution_validation():
    assert not build_cell_template(Polygon()).cells
    with pytest.raises(ValueError):
        build_cell_template(box(0, 0, 1000, 1000), boundary_resolution_mm=0)


def test_area_floor_limits_diagonal_refinement_and_is_part_of_the_cache_key():
    region = Polygon([(0, 0), (3000, 0), (1650, 3000), (0, 3000)])
    original = build_cell_template(region, boundary_resolution_mm=20, min_cell_area_mm2=0)
    limited = build_cell_template(region, boundary_resolution_mm=20, min_cell_area_mm2=40000)
    assert len(limited.cells) < len(original.cells)
    assert limited.excluded_area_mm2 > original.excluded_area_mm2
    assert all(box(*bounds).area >= 40000 for bounds in limited.cells)
    assert all(box(*bounds).covered_by(region) for bounds in limited.cells)
    assert cached_cell_template(region.wkb, (), 20, 0) is not cached_cell_template(region.wkb, (), 20, 40000)


def test_area_floor_applies_to_initial_wall_bands_after_legal_merges():
    region = box(0, 0, 2000, 1000)
    wall = box(950, 0, 960, 1000)
    original = build_cell_template(region, [wall], min_cell_area_mm2=0)
    limited = build_cell_template(region, [wall], min_cell_area_mm2=40000)
    assert len(original.cells) == 3 and len(limited.cells) == 2
    assert limited.discarded_small_cells == 1
    assert limited.excluded_area_mm2 == pytest.approx(10000)
    assert all(box(*bounds).area >= 40000 for bounds in limited.cells)
    with pytest.raises(ValueError):
        build_cell_template(region, min_cell_area_mm2=-1)
