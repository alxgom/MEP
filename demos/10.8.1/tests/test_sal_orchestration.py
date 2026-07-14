import pytest

from mep_routing.installations.sal.orchestration import (
    SalRoutingStrategy,
    dispatch_flow_strategy,
    is_flow_strategy,
    is_negotiated_strategy,
    sequential_room_orders,
    should_stop_after_sequential_candidate,
)
from mep_routing.installations.sal.policy import SalSolverPolicy
from mep_routing.installations.sal.prepared import SalPreparedRoutingProblem
from mep_routing.installations.sal.route_plan import build_sal_route_plan


def test_sequential_room_orders_match_sal_strategy_policy():
    rooms = ("Bathroom", "Bedroom")

    assert sequential_room_orders(SalRoutingStrategy.GREEDY_DUAL_SORT, rooms) == (
        rooms,
        tuple(reversed(rooms)),
    )
    assert sequential_room_orders(SalRoutingStrategy.FIRST_FIT, rooms) == (
        ("Bathroom", "Bedroom"),
        ("Bedroom", "Bathroom"),
    )
    assert sequential_room_orders(SalRoutingStrategy.MIN_COST_FLOW_SMALL_PINS, rooms) == ()
    assert should_stop_after_sequential_candidate(SalRoutingStrategy.FIRST_FIT, 0)
    assert not should_stop_after_sequential_candidate(SalRoutingStrategy.BEST_FIT, 0)


def test_flow_dispatch_preserves_each_strategy_callback_signature():
    prepared = SalPreparedRoutingProblem(
        route_plan=build_sal_route_plan({"Bathroom": (0, 0)}, (0, 0)),
        policy=SalSolverPolicy(100, 5, 1.05, 200, 150, 1e9, 1.05, 0),
        pin_node_map={"tl": [1]},
        global_pins={"tl": (1, 2)},
        shaft_boundary_nodes=[3],
        shaft_node_idx=3,
        chosen_shaft_pin="left_mid",
        chosen_shaft_target={"pin": "left_mid"},
        shaft_path=[4, 5],
    )
    calls = []

    def run_small(*args):
        calls.append(("small", args))
        return "small-result"

    def run_two_stage(*args):
        calls.append(("two-stage", args))
        return "two-stage-result"

    assert dispatch_flow_strategy(5, prepared, run_small_pin_flow=run_small, run_two_stage_flow=run_two_stage) == "small-result"
    assert dispatch_flow_strategy(6, prepared, run_small_pin_flow=run_small, run_two_stage_flow=run_two_stage) == "two-stage-result"
    assert dispatch_flow_strategy(1, prepared, run_small_pin_flow=run_small, run_two_stage_flow=run_two_stage) is None
    assert calls == [
        ("small", (prepared,)),
        ("two-stage", (prepared,)),
    ]


def test_strategy_categories_and_unknown_selector_are_explicit():
    assert is_flow_strategy(SalRoutingStrategy.MIN_COST_FLOW_TWO_STAGE)
    assert not is_flow_strategy(SalRoutingStrategy.BEST_FIT)
    assert is_negotiated_strategy(SalRoutingStrategy.NEGOTIATED_CONGESTION_FAVOUR_LARGE)

    with pytest.raises(ValueError):
        sequential_room_orders(99, ())
