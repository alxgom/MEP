import numpy as np
from shapely.geometry import box

from mep_routing.graphs.lifecycle import GraphLifecycle


def port_specs(pins, _angle):
    return [
        {"pin": name, "out_dir": "E", "access_point": point}
        for name, point in pins.items()
    ]


def make_lifecycle(*, terminals=None):
    return GraphLifecycle(
        routing_region=box(0, 0, 1000, 1000),
        wall_polygons=(),
        covers=(),
        columns=(),
        shafts=(),
        walls=(),
        terminals=terminals or {"bath": (500, 500)},
        shaft_extraction=None,
        shaft_core_entry_specs=(),
        grid_spacing_mm=100,
        scaffold_spacing_mm=400,
        wall_thickness_mm=20,
        wall_clearance_mm=0,
        epsilon_mm=100,
        port_access_specs=port_specs,
    )


def test_commit_runtime_restricts_machine_pin_edges_by_access_direction():
    lifecycle = make_lifecycle()
    runtime = lifecycle.commit_runtime(
        np.asarray(((0, 0), (100, 0), (0, 100)), dtype=np.float32),
        [(0, 1, 100.0, "E"), (0, 2, 100.0, "N")],
        {"supply": (0, 0)},
        0,
    )

    assert runtime.edge_list == [(0, 1, 100.0, "E")]
    assert runtime.adjacency[0] == [(1, 100.0, "E")]


def test_lifecycle_builds_base_regular_and_caches_hannan_static_axes():
    lifecycle = make_lifecycle()

    base = lifecycle.build_base_regular()
    first = lifecycle.hannan_template(shift_walls=True)
    second = lifecycle.hannan_template(shift_walls=True)

    assert base is not None
    assert len(base.nodes) > 0
    assert first is second
    lifecycle.clear_hannan_templates()
    assert lifecycle.hannan_template(shift_walls=True) is not first


def test_lifecycle_epsilon_mode_preserves_machine_access_points():
    lifecycle = make_lifecycle()

    result = lifecycle.build_selected(2, {"supply": (300, 500)}, 0)

    assert result is not None
    assert result.variant is not None
    assert (300.0, 500.0) in {tuple(node) for node in result.runtime.nodes}


def test_dynamic_obstacle_preserves_terminal_node_when_it_matches_graph_node():
    lifecycle = make_lifecycle(terminals={"bath": (10, 0)})
    runtime = lifecycle.commit_runtime(
        np.asarray(((0, 0), (10, 0), (20, 0)), dtype=np.float32),
        [(0, 1, 10.0, "E"), (1, 2, 10.0, "E")],
        {},
        0,
    )

    result = lifecycle.apply_dynamic_obstacle(runtime, box(8, -2, 12, 2), {}, 0, clearance_mm=0)

    assert result.blocked_node_count == 0
    assert result.blocked_edge_count == 0
    assert result.env.adj[1] == [(0, 10.0, "W"), (2, 10.0, "E")]
