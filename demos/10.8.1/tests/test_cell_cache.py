"""Dwelling partitions survive dragging; machine patches stay local."""

import numpy as np
import pytest
from shapely.affinity import rotate
from shapely.geometry import LineString, box
from shapely.ops import unary_union

from mep_routing.graphs.cells import cached_cell_template, cached_machine_cell_template
from mep_routing.graphs.lifecycle import GraphMode
from mep_routing.graphs.variants import build_cell_variant
from test_graph_lifecycle import make_lifecycle


def patch(base, body, region, walls=(), resolution=200, minimum=40000):
    return cached_machine_cell_template(base, body.wkb, tuple(w.wkb for w in walls),
                                        resolution, minimum, region.difference(body).area)


def test_local_patch_preserves_untouched_cells_and_covers_free_space():
    region, wall = box(0, 0, 6000, 4000), box(2900, 0, 3100, 4000)
    base = cached_cell_template(region.wkb, (wall.wkb,), 200)
    body = box(1000, 1500, 2000, 2500)
    result, rebuilt = patch(base, body, region, [wall])
    untouched = [bounds for bounds in base.cells if box(*bounds).intersection(body).area == 0]
    assert len(base.cells) == 3 and rebuilt == 1
    assert all(bounds in result.cells for bounds in untouched)
    geometry = unary_union([box(*bounds) for bounds in result.cells])
    assert geometry.symmetric_difference(region.difference(body)).area < 1e-5
    assert sum(box(*b).area for b in result.cells) == pytest.approx(geometry.area)
    assert any(result.cells[a] in untouched and result.cells[b] not in untouched
               or result.cells[b] in untouched and result.cells[a] not in untouched
               for a, b, *_ in result.portals)
    assert (result, rebuilt) == patch(base, body, region, [wall])
    assert base is cached_cell_template(region.wkb, (wall.wkb,), 200)


def test_covered_cells_are_removed_and_boundary_contact_does_not_rebuild():
    region = box(0, 0, 2000, 1000)
    base = cached_cell_template(region.wkb, (), 200)
    covered, count = patch(base, region.buffer(100), region)
    assert count == 1 and not covered.cells and not covered.portals
    assert covered.excluded_area_mm2 == 0
    for body in (box(2000, 0, 3000, 1000), box(3000, 3000, 4000, 4000)):
        unchanged, count = patch(base, body, region)
        assert count == 0 and unchanged == base


def test_rotated_machine_keeps_area_floor_and_excluded_area_accounting():
    region = box(0, 0, 3000, 3000)
    base = cached_cell_template(region.wkb, (), 20)
    body = rotate(box(900, 900, 2100, 2100), 27)
    result, count = patch(base, body, region, resolution=20)
    assert count == 1 and result.excluded_area_mm2 > 0
    for bounds in result.cells:
        cell = box(*bounds)
        assert cell.area >= 40000 and cell.covered_by(region.difference(body))
    retained_area = sum(box(*bounds).area for bounds in result.cells)
    assert result.excluded_area_mm2 == pytest.approx(region.difference(body).area - retained_area)


def test_dragging_more_poses_than_pose_cache_capacity_reuses_one_dwelling_partition():
    cached_cell_template.cache_clear()
    cached_machine_cell_template.cache_clear()
    region = box(0, 0, 4000, 3000)
    for x in range(30):
        build_cell_variant(allowed_region=region, wall_polys=(), wall_thickness_mm=150,
                           required_points=[(200, 200)], machine_polygon=box(500 + x, 500, 1000 + x, 1000))
    info = cached_cell_template.cache_info()
    assert info.misses == 1 and info.hits == 29 and info.currsize == 1
    assert cached_machine_cell_template.cache_info().currsize == 16


def test_dwelling_geometry_and_partition_settings_invalidate_the_base_cache():
    cached_cell_template.cache_clear()
    region = box(0, 0, 4000, 3000)
    first = cached_cell_template(region.wkb, (), 200, 40000)
    assert first is cached_cell_template(region.wkb, (), 200, 40000)
    for geometry, walls, resolution, minimum in (
        (region.buffer(100), (), 200, 40000),
        (region, (box(1900, 0, 2100, 3000).wkb,), 200, 40000),
        (region, (), 100, 40000), (region, (), 200, 0),
    ):
        assert cached_cell_template(geometry.wkb, walls, resolution, minimum) is not first
    assert cached_cell_template.cache_info().misses == 5


def test_both_cell_modes_use_the_same_static_partition_before_cutting_the_machine():
    cached_cell_template.cache_clear()
    lifecycle = make_lifecycle()
    lifecycle.port_access_specs = lambda pins, angle: [
        {"pin": "supply", "out_dir": "E", "access_point": pins["supply"]}]
    for mode in (GraphMode.CELL_SKELETON, GraphMode.HANNAN_CELL):
        for dx in (0, 100):
            pins = {"c_tl": (100 + dx, 100), "c_tr": (300 + dx, 100),
                    "c_br": (300 + dx, 300), "c_bl": (100 + dx, 300), "supply": (400 + dx, 200)}
            result = lifecycle.build_selected(mode, pins, 0)
            assert result.variant.diagnostics["base_cells"] == 1
            assert result.variant.diagnostics["rebuilt_cells"] == 1
            assert (400 + dx, 200) in set(map(tuple, result.runtime.nodes))
    info = cached_cell_template.cache_info()
    assert info.misses == 1 and info.hits == 3


def test_required_points_change_attachments_without_invalidating_partition_or_patch():
    region, body = box(0, 0, 4000, 3000), box(1000, 1000, 2000, 2000)
    kwargs = dict(allowed_region=region, wall_polys=(), wall_thickness_mm=150, machine_polygon=body)
    first = build_cell_variant(**kwargs, required_points=[(200, 500)])
    before = cached_machine_cell_template.cache_info()
    second = build_cell_variant(**kwargs, required_points=[(3500, 2500)])
    assert cached_machine_cell_template.cache_info().hits == before.hits + 1
    assert first.diagnostics["cells"] == second.diagnostics["cells"]
    assert (3500, 2500) in set(map(tuple, second.nodes))
    for result in (first, second):
        assert all(LineString((result.nodes[u], result.nodes[v])).intersection(body.buffer(-1e-3)).length < 1e-4
                   for u, v, *_ in result.edges)
    assert not np.array_equal(first.nodes, second.nodes)
