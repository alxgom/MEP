import pytest

from mep_routing.installations.sal.route_plan import (
    SAL_LARGE_PORTS,
    SAL_SMALL_PORTS,
    build_sal_route_plan,
)


def test_sal_route_plan_preserves_network_classification_and_distance_order():
    plan = build_sal_route_plan(
        {
            "Kitchen": (2, 0),
            "Bathroom Far": (8, 0),
            "Bedroom": (1, 0),
            "Washroom": (3, 0),
            "Toilet": (5, 0),
        },
        (0, 0),
    )

    assert plan.small_routes == ("Washroom", "Toilet", "Bathroom Far")
    assert plan.large_routes == ("Shaft", "Kitchen")
    assert plan.all_routes == ("Shaft", "Kitchen", "Washroom", "Toilet", "Bathroom Far")
    assert plan.has_kitchen
    assert plan.large_ports == SAL_LARGE_PORTS == ("left_mid", "right_mid")
    assert plan.small_ports == SAL_SMALL_PORTS == ("tl", "tr", "bl", "br")


def test_sal_route_plan_keeps_missing_kitchen_explicit_and_maps_opposite_port():
    plan = build_sal_route_plan({"Bathroom": (1, 1)}, (0, 0))

    assert not plan.has_kitchen
    assert plan.large_routes == ("Shaft", "Kitchen")
    assert plan.kitchen_port_for("left_mid") == "right_mid"
    assert plan.kitchen_port_for("right_mid") == "left_mid"
    with pytest.raises(ValueError, match="Unknown Sal large port"):
        plan.kitchen_port_for("tl")
