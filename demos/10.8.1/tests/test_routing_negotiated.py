from types import SimpleNamespace

from mep_routing.routing.negotiated import (
    NegotiatedProblem,
    NegotiatedRuntime,
    solve_negotiated,
)


def _runtime(run_search, count_crossings):
    return NegotiatedRuntime(
        env=SimpleNamespace(adj={
            0: [(1, 10.0, "E")],
            1: [(0, 10.0, "W")],
        }),
        set_terminal_block_weight=lambda *_args: None,
        add_route_clearance_weights=lambda *_args: None,
        add_route_interaction_weights=lambda *_args: None,
        route_diameter=lambda _name: 100,
        route_segments_from_path=lambda name, path, *_args: [(name, tuple(path))],
        route_axis_records=lambda *_args: [],
        run_search=run_search,
        count_crossings=count_crossings,
        score_routes=lambda routes, crossings: len(routes) + crossings,
    )


def test_negotiated_solver_uses_current_assignments_and_exits_crossing_free():
    eligibility_calls = []
    search_calls = []

    def eligible(name, assignments):
        eligibility_calls.append((name, dict(assignments)))
        return ("out",) if name == "Core" else ("return",)

    def run_search(_env, starts, pins, *_args, **_kwargs):
        search_calls.append((tuple(starts), tuple(pins)))
        return [0, 1], 10.0, pins[0], {"pin": pins[0]}

    problem = NegotiatedProblem(
        network_names=("Core", "Return"),
        start_nodes=lambda name: [0] if name == "Core" else [1],
        eligible_pins=eligible,
        terminal_nodes={},
        congestion_scale=lambda _name: 1.0,
    )
    result = solve_negotiated(
        problem,
        _runtime(run_search, lambda _routes: 0),
        {},
        {},
        machine_angle=0,
        bend_cost=100,
    )

    assert result.success and result.crossing_free and result.attempts == 1
    assert [name for name, _segments in result.routes] == ["Core", "Return"]
    assert eligibility_calls == [
        ("Core", {}),
        ("Return", {"Core": "out"}),
    ]
    assert search_calls == [((0,), ("out",)), ((1,), ("return",))]


def test_negotiated_solver_applies_scale_and_history_after_conflicting_candidate():
    observed_weights = []
    crossing_results = iter((1, 0))

    def run_search(_env, _starts, pins, *_args, **kwargs):
        observed_weights.append((pins[0], dict(kwargs["edge_weights"])))
        return [0, 1], 10.0, pins[0], {"pin": pins[0]}

    problem = NegotiatedProblem(
        network_names=("Supply", "Return"),
        start_nodes=lambda _name: [0],
        eligible_pins=lambda name, _assignments: (name.lower(),),
        terminal_nodes={},
        congestion_scale=lambda name: 0.5 if name == "Return" else 1.0,
    )
    result = solve_negotiated(
        problem,
        _runtime(run_search, lambda _routes: next(crossing_results)),
        {},
        {},
        machine_angle=0,
        bend_cost=100,
        iterations=2,
        present_penalty=100.0,
        history_penalty=40.0,
    )

    first_supply = observed_weights[0][1][(0, 1)]
    first_return = observed_weights[1][1][(0, 1)]
    second_supply = observed_weights[2][1][(0, 1)]
    assert first_supply == 10.0
    assert first_return == 110.0  # two present-use contributions, scaled by 0.5
    assert second_supply == 290.0  # present use plus edge and node history
    assert result.success and result.crossing_free and result.attempts == 2


def test_negotiated_solver_returns_failure_when_a_network_never_routes():
    problem = NegotiatedProblem(
        network_names=("Supply", "Return"),
        start_nodes=lambda name: [] if name == "Return" else [0],
        eligible_pins=lambda name, _assignments: (name,),
        terminal_nodes={},
        congestion_scale=lambda _name: 1.0,
    )

    result = solve_negotiated(
        problem,
        _runtime(
            lambda _env, _starts, pins, *_args, **_kwargs: (
                [0, 1], 10.0, pins[0], {"pin": pins[0]},
            ),
            lambda _routes: 0,
        ),
        {},
        {},
        machine_angle=0,
        bend_cost=100,
        iterations=3,
    )

    assert not result.success
    assert result.routes is None
    assert result.attempts == 3
