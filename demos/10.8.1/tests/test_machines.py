from mep_routing.domain import (
    machine_pins,
    outward_vector,
    port_access_specs,
)
from mep_routing.installations.sal import (
    KITCHEN_ROUTE_NAME,
    LARGE_DUCT_ROUTE_NAMES,
    SAL_OZEO_FLAT_MACHINE,
    SHAFT_ROUTE_NAME,
)


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


def test_machine_pins_match_baseline_unrotated_geometry():
    pins = machine_pins(SAL_OZEO_FLAT_MACHINE, 0, 0, 0)

    assert pins["left_mid"] == (-256, 0)
    assert pins["right_mid"] == (256, 0)
    assert pins["tl"] == (-256, 185)
    assert pins["br"] == (256, -185)
    assert pins["c_tl"] == (-256, 230)
    assert pins["c_br"] == (256, -230)


def test_port_access_specs_match_baseline_stub_policy():
    pins = machine_pins(SAL_OZEO_FLAT_MACHINE, 0, 0, 0)
    specs = port_access_specs(SAL_OZEO_FLAT_MACHINE, pins, 0)
    by_pin = {}
    for spec in specs:
        by_pin.setdefault(spec["pin"], []).append(spec)

    assert by_pin["left_mid"] == [{
        "pin": "left_mid",
        "pin_point": (-256.0, 0.0),
        "access_point": (-506, 0),
        "out_dir": "W",
        "in_dir": "E",
    }]
    assert {
        (spec["access_point"], spec["out_dir"], spec["in_dir"])
        for spec in by_pin["tl"]
    } == {
        ((-356, 185), "W", "E"),
        ((-256, 285), "N", "S"),
    }


def test_outward_vector_matches_pin_side_and_rotation():
    assert outward_vector("left_mid", 0) == "W"
    assert outward_vector("right_mid", 0) == "E"
    assert outward_vector("left_mid", 90) == "S"
