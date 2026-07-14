from types import SimpleNamespace

from mep_routing.installations.sal.flow_runtime import SalFlowRuntime
from mep_routing.installations.sal.policy import SalSolverPolicy
from mep_routing.installations.sal.route_plan import build_sal_route_plan


def test_small_pin_flow_returns_empty_result_without_route_names():
    runtime = SalFlowRuntime(
        env=SimpleNamespace(adj={}), route_plan=build_sal_route_plan({}, (0, 0)),
        terminals={}, small_diameter=90, large_diameter=125,
        policy=SalSolverPolicy(100, 5, 1.05, 200, 150, 1e9, 1.05, 0), source_start_nodes=lambda value: value,
        weighted_edge_cost=lambda *_args: 0, line_graph_direction=lambda *_args: 0,
        record_edge_weight_overlay=lambda *_args: None, route_start_nodes=lambda _name: [],
        route_segments_from_path=lambda *_args: [], build_routes_from_paths=lambda *_args: ([], 0),
        route_axis_records=lambda *_args: [], add_static_clearance_weights=lambda *_args, **_kwargs: None,
        add_machine_clearance_weights=lambda *_args: None, add_route_clearance_weights=lambda *_args: None,
        add_route_interaction_weights=lambda *_args: None, route_diameter=lambda _name: 90,
        run_search=lambda *_args, **_kwargs: (None, None, None, None), count_crossings=lambda _routes: 0,
        score_routes=lambda _routes, _crossings: 0,
    )

    assert runtime.run_pin_flow([], {}, {}) == ({}, {}, 0.0, 0)
