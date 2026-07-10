from vent_router.domain import SAL_OZEO_FLAT_MACHINE


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

    assert spec.route_diameter_mm("Shaft") == 125
    assert spec.route_diameter_mm("Kitchen") == 125
    assert spec.route_diameter_mm("Bathroom") == 90


def test_sal_ozeo_flat_pin_stub_policy_matches_baseline():
    spec = SAL_OZEO_FLAT_MACHINE

    assert spec.pin_stub_length_mm("left_mid") == 250
    assert spec.pin_stub_length_mm("right_mid") == 250
    assert spec.pin_stub_length_mm("tl") == 100

