from types import SimpleNamespace

import mep_routing.installations.sal.flow_runtime as flow_runtime_module
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


def test_pin_flow_runtime_preserves_tuple_success_and_weight_overlay():
    overlays = []
    runtime = _runtime(
        SimpleNamespace(adj={
            0: [(1, 10.0, "H")],
            1: [(0, 10.0, "H")],
        }),
        overlays,
    )
    target = {"pin": "tl", "node_idx": 1, "in_dir": "H"}

    result = runtime.run_pin_flow(
        ["Bathroom"],
        {"Bathroom": [target]},
        {"Bathroom": [0]},
        edge_weights={(0, 1): 4.0},
    )

    assert result == (
        {"Bathroom": [0, 1]},
        {"Bathroom": target},
        4.0,
        1,
    )
    assert overlays == [({(0, 1): 4.0}, runtime.env)]


def test_pin_flow_runtime_preserves_partial_flow_failure_count():
    runtime = _runtime(SimpleNamespace(adj={0: [], 1: []}), [])
    target = {"pin": "tl", "node_idx": 1, "in_dir": "H"}

    assert runtime.run_pin_flow(
        ["Bathroom"], {"Bathroom": [target]}, {"Bathroom": [0]}
    ) == (None, None, 0.0, 0)
    assert runtime.run_pin_flow(
        ["Bathroom"], {"Bathroom": []}, {"Bathroom": [0]}
    ) == (None, None, float("inf"), 0)
    assert runtime.run_pin_flow(
        ["Bathroom"], {"Bathroom": [target]}, {"Bathroom": []}
    ) == (None, None, 0.0, 0)


def test_prepared_sequential_uses_runtime_callbacks(monkeypatch):
    runtime = _runtime(SimpleNamespace(adj={}), [])
    runtime.terminal_node_indices = lambda *_args: {"Bathroom": 4}
    runtime.set_terminal_block_weight = lambda *_args: None
    prepared = _prepared(runtime)
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "sequential-result"

    monkeypatch.setattr(flow_runtime_module, "run_sequential_routing", fake_run)

    assert runtime.run_prepared_sequential(
        prepared, ("Bathroom",), machine_angle=90
    ) == "sequential-result"
    assert captured["args"][0] == ("Bathroom",)
    assert captured["kwargs"]["route_plan"] is prepared.route_plan
    assert captured["kwargs"]["terminal_node_indices"] is runtime.terminal_node_indices
    assert captured["kwargs"]["set_terminal_block_weight"] is runtime.set_terminal_block_weight
    assert captured["kwargs"]["machine_angle"] == 90


def test_prepared_negotiated_builds_context_from_runtime(monkeypatch):
    runtime = _runtime(SimpleNamespace(adj={}), [])
    runtime.terminal_node_indices = lambda *_args: {"Shaft": 4}
    runtime.set_terminal_block_weight = lambda *_args: None
    prepared = _prepared(runtime)
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "negotiated-result"

    monkeypatch.setattr(flow_runtime_module, "run_negotiated_congestion", fake_run)

    assert runtime.run_prepared_negotiated(
        prepared, True, machine_angle=180
    ) == "negotiated-result"
    context = captured["kwargs"]["context"]
    assert context.env is runtime.env
    assert context.terminal_node_indices is runtime.terminal_node_indices
    assert context.set_terminal_block_weight is runtime.set_terminal_block_weight
    assert captured["kwargs"]["favour_large"] is True
    assert captured["kwargs"]["machine_angle"] == 180


def _runtime(env, overlays):
    return SalFlowRuntime(
        env=env, route_plan=build_sal_route_plan({}, (0, 0)),
        terminals={}, small_diameter=90, large_diameter=125,
        policy=SalSolverPolicy(100, 5, 1.05, 200, 150, 1e9, 1.05, 0),
        source_start_nodes=lambda value: value,
        weighted_edge_cost=lambda weights, u, v, distance: weights.get((u, v), distance),
        line_graph_direction=lambda _env, _u, _v: "H",
        record_edge_weight_overlay=lambda weights, active_env: overlays.append((weights, active_env)),
        route_start_nodes=lambda _name: [], route_segments_from_path=lambda *_args: [],
        build_routes_from_paths=lambda *_args: ([], 0), route_axis_records=lambda *_args: [],
        add_static_clearance_weights=lambda *_args, **_kwargs: None,
        add_machine_clearance_weights=lambda *_args: None, add_route_clearance_weights=lambda *_args: None,
        add_route_interaction_weights=lambda *_args: None, route_diameter=lambda _name: 90,
        run_search=lambda *_args, **_kwargs: (None, None, None, None), count_crossings=lambda _routes: 0,
        score_routes=lambda _routes, _crossings: 0,
    )


def _prepared(runtime):
    return SimpleNamespace(
        route_plan=runtime.route_plan,
        policy=runtime.policy,
        pin_node_map={"tl": []},
        global_pins={"tl": (0, 0)},
        shaft_boundary_nodes=(0,),
        shaft_node_idx=0,
        shaft_path=[0],
        chosen_shaft_pin="tl",
        chosen_shaft_target={"pin": "tl"},
    )
