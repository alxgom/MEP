from types import SimpleNamespace

from mep_routing.installations.sal.application import (
    SalApplicationAdapter,
    SalApplicationHooks,
    SalSolverSettings,
)


def _settings():
    return SalSolverSettings(
        bend_cost=1200,
        crossing_penalty_multiplier=2.5,
        duct_buffer_ratio=0.25,
        shaft_clearance_mm=100,
        machine_clearance_soft_margin_mm=50,
        overlap_block_weight=1e9,
        min_piece_factor=1.1,
        heuristic_mode=2,
        routing_strategy=3,
    )


def test_sal_application_settings_build_one_policy_snapshot():
    policy = _settings().policy()
    assert policy.bend_cost == 1200
    assert policy.crossing_penalty == 3000
    assert policy.heuristic_mode == 2


def test_sal_application_preflight_stops_before_live_graph_adapters():
    calls = []
    installation = SimpleNamespace(
        build_route_plan=lambda terminals, center: calls.append((terminals, center)) or SimpleNamespace(),
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("live graph hook should not run after failed preflight")

    hooks = SalApplicationHooks(
        preflight_error=lambda: "Blocked: test",
        grid_available=unexpected,
        machine_pins=unexpected,
        refresh_graph=unexpected,
        current_env=unexpected,
        snap_pins=unexpected,
        shaft_entry_nodes=unexpected,
        terminal_nodes=unexpected,
        block_terminal_edges=unexpected,
        routing_runtime=unexpected,
        source_start_nodes=unexpected,
        weighted_edge_cost=unexpected,
        line_graph_direction=unexpected,
        route_start_nodes=unexpected,
        route_segments_from_path=unexpected,
        build_routes_from_paths=unexpected,
        route_diameter=unexpected,
        count_crossings=unexpected,
        score_routes=unexpected,
        conflict_summary=unexpected,
    )
    result = SalApplicationAdapter(
        installation=installation,
        terminals={"Kitchen": (10, 20)},
        machine_center=(30, 40),
        machine_angle=0,
        small_diameter=100,
        large_diameter=200,
        settings=_settings(),
        hooks=hooks,
    ).solve()

    assert calls == [({"Kitchen": (10, 20)}, (30, 40))]
    assert result.routes is None
    assert result.status.startswith("Blocked: test")
