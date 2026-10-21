"""Cell enrichment reuses Hanan wall-shift axes and preserves valid topology."""

from shapely.geometry import LineString, Point, box

from mep_routing.graphs.cells import cached_cell_template
from mep_routing.graphs.hannan import build_static_axes, build_wall_shift_axes
from mep_routing.graphs.runtime import create_runtime_graph
from mep_routing.graphs.variants import build_cell_variant
from test_cell_skeleton import reachable


def test_wall_axes_use_normal_offset_midpoints_and_integer_millimetres():
    horizontal = LineString([(500, 1000), (3500, 1000)])
    assert build_wall_shift_axes([horizontal], 100) == ({2000}, {900, 1100})
    diagonal = LineString([(100, 200), (900, 800)])  # 800/600/1000 triangle.
    assert build_wall_shift_axes([diagonal], 100) == ({440, 560}, {420, 580})
    assert build_wall_shift_axes([LineString([(0, 0), (0.1, 0.1)])], 100) == (set(), set())


def test_existing_hanan_static_axes_preserve_wall_priorities_and_other_sources():
    walls = [LineString([(100, 200), (900, 800)]), LineString([(1000, 100), (1000, 1900)])]
    kwargs = dict(allowed_region=box(0, 0, 4000, 2000), terminals={"t": (333, 777)},
                  shaft_extraction=None, covers=(), columns=(), shafts=(), wall_polys=(), walls=walls,
                  grid_spacing_mm=200, scaffold_spacing_mm=600, wall_clearance_mm=100)
    base = build_static_axes(**kwargs, shift_walls=False)
    shifted = build_static_axes(**kwargs, shift_walls=True)
    wall_x, wall_y = {440, 560, 900, 1100}, {420, 580, 1000}
    assert shifted["xs"] == base["xs"] | wall_x
    assert shifted["ys"] == base["ys"] | wall_y
    assert shifted["priority_x"] == base["priority_x"] | wall_x
    assert shifted["priority_y"] == base["priority_y"] | wall_y
    assert shifted["preserve_x"] == base["preserve_x"]
    assert shifted["preserve_y"] == base["preserve_y"]


def test_cell_wall_axes_merge_close_coordinates_and_connect_to_cell_centre():
    region = box(0, 0, 4000, 2000)
    template = cached_cell_template(region.wkb, (), 200, 40000)
    before = template.cells
    result = build_cell_variant(
        allowed_region=region, wall_polys=(), wall_thickness_mm=150, required_points=[],
        walls=[LineString([(500, 1000), (3500, 1000)]), LineString([(560, 1000), (3560, 1000)])],
        wall_offset_mm=100,
    )
    assert result.axes_x == [2000] and result.axes_y == [900, 1100]
    assert result.diagnostics["wall_nodes"] == 2
    runtime = create_runtime_graph(result.nodes, result.edges)
    centre = next(i for i, p in enumerate(result.nodes) if tuple(p) == (2000, 1000))
    assert len(reachable(runtime.adjacency, centre)) == len(result.nodes)
    assert template.cells == before


def test_wall_axes_do_not_bridge_disconnected_regions_or_obstacles():
    region = box(0, 0, 4000, 2000).difference(box(1800, 0, 2200, 2000))
    result = build_cell_variant(
        allowed_region=region, wall_polys=(), wall_thickness_mm=150, required_points=[],
        walls=[LineString([(200, 1000), (800, 1000)]), LineString([(3200, 1000), (3800, 1000)])],
    )
    runtime = create_runtime_graph(result.nodes, result.edges)
    index = {tuple(p): i for i, p in enumerate(result.nodes)}
    assert index[(3500, 900)] not in reachable(runtime.adjacency, index[(500, 900)])
    assert all(LineString((result.nodes[u], result.nodes[v])).covered_by(region) for u, v, *_ in result.edges)


def test_wall_axes_and_cell_crossings_share_nodes_and_machine_interior_is_excluded():
    region, body = box(0, 0, 4000, 2000), box(1800, 800, 2200, 1200)
    result = build_cell_variant(
        allowed_region=region, wall_polys=(), wall_thickness_mm=150,
        required_points=[(300, 300), (3700, 1700)], machine_polygon=body,
        walls=[LineString([(100, 600), (2900, 600)]), LineString([(1000, 1600), (4000, 1600)])],
        wall_offset_mm=100,
    )
    assert result.diagnostics["wall_nodes"]
    assert all((c - a) * (d - b) >= 40000 for a, b, c, d in result.diagnostics["cells"])
    for index, (u, v, _, direction) in enumerate(result.edges):
        assert direction in ("E", "W", "N", "S")
        line = LineString((result.nodes[u], result.nodes[v]))
        assert line.intersection(body.buffer(-1e-3)).length < 1e-4
        for a, b, *_ in result.edges[index + 1:]:
            intersection = line.intersection(LineString((result.nodes[a], result.nodes[b])))
            if not intersection.is_empty:
                assert isinstance(intersection, Point) and {u, v} & {a, b}


def test_no_walls_add_no_axes_or_wall_grid():
    result = build_cell_variant(allowed_region=box(0, 0, 1000, 1000), wall_polys=(), wall_thickness_mm=150,
                                required_points=[(100, 100)])
    assert not result.axes_x and not result.axes_y
    assert result.diagnostics["wall_nodes"] == result.diagnostics["wall_links"] == 0
    assert result.diagnostics["attached"] == 1
    assert result.diagnostics["min_cell_area_mm2"] == 40000


def test_required_point_can_attach_to_wall_axes_when_area_floor_removes_its_cell():
    result = build_cell_variant(
        allowed_region=box(0, 0, 1000, 300), wall_polys=(), wall_thickness_mm=150,
        required_points=[(333, 160)], min_cell_area_mm2=400000,
        walls=[LineString([(100, 150), (900, 150)])], wall_offset_mm=100,
    )
    assert not result.diagnostics["cells"] and result.diagnostics["endpoint_links"] > 0
    runtime = create_runtime_graph(result.nodes, result.edges)
    endpoint = next(i for i, p in enumerate(result.nodes) if tuple(p) == (333, 160))
    assert runtime.adjacency[endpoint]
    assert result.diagnostics["attached"] == 1


def test_isolated_cell_centre_projects_onto_a_wall_edge_without_vertex_alignment():
    region = box(0, 0, 1800, 2000)
    result = build_cell_variant(
        allowed_region=region, wall_polys=(), wall_thickness_mm=150, required_points=[],
        walls=[LineString([(200, 1000), (800, 1000)])],
    )
    runtime = create_runtime_graph(result.nodes, result.edges)
    index = {tuple(p): i for i, p in enumerate(result.nodes)}
    assert len(reachable(runtime.adjacency, index[(900, 1000)])) == len(result.nodes)
    assert (500, 1000) in index  # Explicit wall-edge attachment, 400 mm from centre.
