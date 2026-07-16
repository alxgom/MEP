import pytest
from shapely.geometry import box

from mep_routing.graphs import GraphMode, SimpleGraphInputs, build_simple_grid
from mep_routing.installations.sal.graph_inputs import build_sal_simple_graph_inputs

from test_graph_lifecycle import make_lifecycle


def test_simple_grid_preserves_required_nodes_and_penalizes_obstacle_edges():
    result = build_simple_grid(SimpleGraphInputs(
        allowed_region=box(0, 0, 1000, 1000),
        required_points=((100, 500), (900, 500)),
        obstacles=(box(450, 400, 550, 600),),
        epsilon_mm=100,
        axis_clearance_mm=100,
    ))

    nodes = {tuple(node) for node in result.nodes}
    assert (100.0, 500.0) in nodes
    assert (900.0, 500.0) in nodes
    assert any(weight == 1e6 for _source, _target, weight, _direction in result.edges)


def test_sal_simple_inputs_use_sal_terminals_access_points_and_fixed_obstacles():
    obstacle = box(400, 400, 500, 500)
    inputs = build_sal_simple_graph_inputs(
        routing_region=box(0, 0, 1000, 1000), terminals={"bath": (100, 100)},
        columns=(obstacle,), shafts=(), machine_pins={}, access_points=((800, 800),),
        epsilon_mm=100, axis_clearance_mm=80,
    )

    assert inputs.required_points == ((100, 100), (800, 800))
    assert inputs.obstacles == (obstacle,)
    assert inputs.axis_clearance_mm == 80


def test_lifecycle_dispatches_named_simple_mode_and_rejects_unknown_mode():
    lifecycle = make_lifecycle()
    lifecycle.simple_input_builder = build_sal_simple_graph_inputs
    result = lifecycle.build_selected(GraphMode.CORE_SIMPLE, {"supply": (300, 500)}, 0)

    assert result.graph_type == int(GraphMode.CORE_SIMPLE)
    assert result.variant is not None
    assert (300.0, 500.0) in {tuple(node) for node in result.runtime.nodes}
    with pytest.raises(ValueError, match="unsupported graph mode"):
        lifecycle.build_selected(99, {}, 0)
