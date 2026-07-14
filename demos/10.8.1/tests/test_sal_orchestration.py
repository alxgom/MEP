import pytest

from mep_routing.installations.sal.orchestration import (
    SalRoutingStrategy,
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


def test_unknown_selector_is_explicit():
    with pytest.raises(ValueError):
        sequential_room_orders(99, ())
