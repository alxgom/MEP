from vent_router.domain import SAL_OZEO_FLAT_MACHINE
from vent_router.domain.routes import KITCHEN_ROUTE_NAME, LARGE_DUCT_ROUTE_NAMES, SHAFT_ROUTE_NAME


def test_sal_ozeo_flat_machine_dimensions_match_demo_baseline():
    spec = SAL_OZEO_FLAT_MACHINE

    assert spec.body_width_mm == 410
    assert spec.body_height_mm == 460
    assert spec.overall_width_mm == 511
    assert spec.small_duct_diameter_mm == 90
    assert spec.large_duct_diameter_mm == 125
    assert spec.small_pin_stub_length_mm == 100
    assert spec.large_pin_stub_length_mm == 250


def test_sal_ozeo_flat_route_diameter_policy_matches_baseline():
    spec = SAL_OZEO_FLAT_MACHINE

    assert spec.route_diameter_mm(SHAFT_ROUTE_NAME) == 125
    assert spec.route_diameter_mm(KITCHEN_ROUTE_NAME) == 125
    assert spec.route_diameter_mm("Bathroom") == 90


def test_sal_ozeo_flat_pin_stub_policy_matches_baseline():
    spec = SAL_OZEO_FLAT_MACHINE

    assert spec.pin_stub_length_mm("left_mid") == 250
    assert spec.pin_stub_length_mm("right_mid") == 250
    assert spec.pin_stub_length_mm("tl") == 100


def test_large_duct_route_names_are_canonical_for_sal_machine():
    assert SAL_OZEO_FLAT_MACHINE.large_route_names is LARGE_DUCT_ROUTE_NAMES
