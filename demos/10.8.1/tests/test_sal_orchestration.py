import pytest

from mep_routing.installations.sal.orchestration import (
    SalFlowRoutingRequest,
    SalRoutingStrategy,
    dispatch_flow_strategy,
    is_flow_strategy,
    is_negotiated_strategy,
    sequential_room_orders,
    should_stop_after_sequential_candidate,
)


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
    request = SalFlowRoutingRequest(
        room_names=("Bathroom",),
        pin_node_map={"tl": [1]},
        global_pins={"tl": (1, 2)},
        shaft_node_idx=3,
        chosen_exhaust_pin="left_mid",
        chosen_exhaust_target={"pin": "left_mid"},
        shaft_path=[4, 5],
    )
    calls = []

    def run_small(*args):
        calls.append(("small", args))
        return "small-result"

    def run_two_stage(*args):
        calls.append(("two-stage", args))
        return "two-stage-result"

    assert dispatch_flow_strategy(5, request, run_small_pin_flow=run_small, run_two_stage_flow=run_two_stage) == "small-result"
    assert dispatch_flow_strategy(6, request, run_small_pin_flow=run_small, run_two_stage_flow=run_two_stage) == "two-stage-result"
    assert dispatch_flow_strategy(1, request, run_small_pin_flow=run_small, run_two_stage_flow=run_two_stage) is None
    assert calls == [
        ("small", (("Bathroom",), {"tl": [1]}, {"tl": (1, 2)}, 3, "left_mid", {"pin": "left_mid"}, [4, 5])),
        ("two-stage", (("Bathroom",), {"tl": [1]}, {"tl": (1, 2)}, [4, 5])),
    ]


def test_strategy_categories_and_unknown_selector_are_explicit():
    assert is_flow_strategy(SalRoutingStrategy.MIN_COST_FLOW_TWO_STAGE)
    assert not is_flow_strategy(SalRoutingStrategy.BEST_FIT)
    assert is_negotiated_strategy(SalRoutingStrategy.NEGOTIATED_CONGESTION_FAVOUR_LARGE)

    with pytest.raises(ValueError):
        sequential_room_orders(99, ())
