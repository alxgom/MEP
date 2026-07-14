from types import SimpleNamespace

import pytest

from mep_routing.installations.sal.orchestration import SalRoutingStrategy
from mep_routing.installations.sal.policy import SalSolverPolicy
from mep_routing.installations.sal.prepared import SalPreparedRoutingProblem
from mep_routing.installations.sal.route_plan import build_sal_route_plan
from mep_routing.installations.sal.strategy_dispatch import (
    SalStrategyRuntime,
    solve_prepared_strategy,
)


def _prepared(room_names=("Bathroom", "Washroom")):
    terminals = {name: (index + 1, 0) for index, name in enumerate(room_names)}
    terminals["Kitchen"] = (10, 0)
    return SalPreparedRoutingProblem(
        route_plan=build_sal_route_plan(terminals, (0, 0)),
        policy=SalSolverPolicy(100, 5, 1.05, 200, 150, 1e9, 1.05, 0),
        global_pins={"left_mid": (0, 0)},
        pin_node_map={"left_mid": []},
        shaft_boundary_nodes=(1,),
        shaft_node_idx=1,
        shaft_path=[1, 2],
        chosen_shaft_pin="left_mid",
        chosen_shaft_target={"pin": "left_mid"},
    )


def _runtime(**overrides):
    values = dict(
        run_small_pin_flow=lambda _prepared: (False, None, "unused", 0),
        run_two_stage_flow=lambda _prepared: (False, None, "unused", 0),
        run_negotiated=lambda _prepared, _favour: SimpleNamespace(
            success=False, routes=None, attempts=4, total_nodes=0,
        ),
        run_sequential=lambda _prepared, _order: (False, None, "unused", 0),
        count_crossings=lambda _routes: 0,
        score_routes=lambda _routes, _crossings: 0.0,
        conflict_summary=lambda _routes: "0 crossings",
    )
    values.update(overrides)
    return SalStrategyRuntime(**values)


@pytest.mark.parametrize(
    ("strategy", "runner_name", "runner_result", "expected_status"),
    (
        (
            SalRoutingStrategy.MIN_COST_FLOW_SMALL_PINS,
            "run_small_pin_flow",
            (True, [("small", [])], "ok", 3),
            "Success: Min-cost flow small pins (0 crossings)",
        ),
        (
            SalRoutingStrategy.MIN_COST_FLOW_TWO_STAGE,
            "run_two_stage_flow",
            (True, [("two", [])], "big-first", 5),
            "Success: Two-stage MCMF big-first (0 crossings)",
        ),
    ),
)
def test_flow_dispatch_preserves_success_status(strategy, runner_name, runner_result, expected_status):
    outcome = solve_prepared_strategy(
        strategy,
        _prepared(),
        _runtime(**{runner_name: lambda _prepared: runner_result}),
    )

    assert outcome.success
    assert outcome.status == expected_status
    assert outcome.total_nodes == runner_result[3]


@pytest.mark.parametrize(
    ("strategy", "runner_name"),
    (
        (SalRoutingStrategy.MIN_COST_FLOW_SMALL_PINS, "run_small_pin_flow"),
        (SalRoutingStrategy.MIN_COST_FLOW_TWO_STAGE, "run_two_stage_flow"),
    ),
)
def test_flow_dispatch_preserves_blocked_status(strategy, runner_name):
    outcome = solve_prepared_strategy(
        strategy,
        _prepared(),
        _runtime(**{runner_name: lambda _prepared: (False, None, "no path", 9)}),
    )

    assert not outcome.success
    assert outcome.routes is None
    assert outcome.total_nodes == 0
    assert outcome.status == "Routing Blocked: no path"


@pytest.mark.parametrize(
    ("strategy", "expected_favour"),
    (
        (SalRoutingStrategy.NEGOTIATED_CONGESTION, False),
        (SalRoutingStrategy.NEGOTIATED_CONGESTION_FAVOUR_LARGE, True),
    ),
)
def test_negotiated_dispatch_preserves_favour_flag_and_attempt_status(strategy, expected_favour):
    calls = []
    routes = [("negotiated", [])]

    def run_negotiated(prepared, favour_large):
        calls.append((prepared, favour_large))
        return SimpleNamespace(success=True, routes=routes, attempts=6, total_nodes=8)

    prepared = _prepared()
    outcome = solve_prepared_strategy(
        strategy,
        prepared,
        _runtime(run_negotiated=run_negotiated),
    )

    assert calls == [(prepared, expected_favour)]
    assert outcome.status == "Success: Routed all (tried 6 iters, 0 crossings)"
    assert outcome.total_nodes == 8


def test_negotiated_dispatch_preserves_blocked_attempt_count():
    outcome = solve_prepared_strategy(
        SalRoutingStrategy.NEGOTIATED_CONGESTION,
        _prepared(),
        _runtime(run_negotiated=lambda *_args: SimpleNamespace(
            success=False, routes=None, attempts=7, total_nodes=0,
        )),
    )

    assert not outcome.success
    assert outcome.status == "Routing Blocked (tried 7 iters)"


def test_best_fit_keeps_lowest_scoring_sequential_candidate():
    calls = []

    def run_sequential(_prepared, order):
        calls.append(order)
        return True, [("/".join(order), [])], "ok", len(order)

    outcome = solve_prepared_strategy(
        SalRoutingStrategy.BEST_FIT,
        _prepared(),
        _runtime(
            run_sequential=run_sequential,
            count_crossings=lambda _routes: 1,
            score_routes=lambda routes, _crossings: 10 if routes[0][0].startswith("Bathroom") else 2,
        ),
    )

    assert calls == [("Bathroom", "Washroom"), ("Washroom", "Bathroom")]
    assert outcome.routes == [("Washroom/Bathroom", [])]
    assert outcome.status == "Success: Routed all (tried 2 perms, 0 crossings)"
    assert outcome.total_nodes == 2


def test_first_fit_stops_on_first_crossing_free_candidate():
    calls = []

    def run_sequential(_prepared, order):
        calls.append(order)
        return True, [("first", [])], "ok", 2

    outcome = solve_prepared_strategy(
        SalRoutingStrategy.FIRST_FIT,
        _prepared(),
        _runtime(run_sequential=run_sequential),
    )

    assert calls == [("Bathroom", "Washroom")]
    assert outcome.success
    assert "tried 1 perms" in outcome.status


def test_sequential_failure_preserves_attempt_count():
    outcome = solve_prepared_strategy(
        SalRoutingStrategy.BEST_FIT,
        _prepared(),
        _runtime(),
    )

    assert not outcome.success
    assert outcome.status == "Routing Blocked (tried 2 perms)"
