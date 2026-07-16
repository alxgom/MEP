from shapely.geometry import box

from mep_routing.domain import MachineSpec
from mep_routing.installations.sal import SalRouteAnalysis, SalSolverPolicy


def _analysis(routing_region=None):
    machine = MachineSpec(
        name="Alternate",
        body_width_mm=300,
        body_height_mm=200,
        overall_width_mm=350,
        small_duct_diameter_mm=80,
        large_duct_diameter_mm=140,
        small_pin_stub_length_mm=50,
        large_pin_stub_length_mm=100,
        large_route_names=frozenset({"Shaft"}),
    )
    policy = SalSolverPolicy(
        bend_cost=100,
        crossing_penalty_multiplier=3,
        duct_buffer_ratio=0.25,
        shaft_clearance_mm=100,
        machine_clearance_soft_margin_mm=50,
        overlap_block_weight=1e9,
        min_piece_factor=1.5,
        heuristic_mode=0,
    )
    return SalRouteAnalysis(machine, policy, routing_region)


def test_route_analysis_uses_selected_machine_and_policy_geometry():
    analysis = _analysis()

    assert analysis.route_diameter("Shaft") == 140
    assert analysis.route_diameter("Bathroom") == 80
    assert analysis.buffered_radius(140) == 18
    assert analysis.required_clearance(80, 140) == 28
    assert analysis.min_piece_length("Shaft") == 420
    assert analysis.min_piece_length("Shaft", terminal_segment=True) == 210


def test_route_analysis_score_uses_policy_derived_crossing_weight():
    routes = [("Shaft", [((0, 0), (1000, 0))])]

    assert _analysis().score(routes, crossings=2) == 1600


def test_route_analysis_reports_quality_and_allowed_region_warnings():
    routes = [
        ("Shaft", [((0, 0), (1000, 0))]),
        ("Bathroom", [((500, -500), (500, 500))]),
    ]

    warnings = _analysis(box(-100, -100, 600, 100)).quality_warnings(routes)

    assert "1 crossing(s)" in warnings
    assert "2 segment(s) outside allowed" in warnings
    assert _analysis().conflict_summary(routes).startswith("1 crossings")
