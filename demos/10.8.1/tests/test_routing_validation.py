from shapely.geometry import Polygon, box

from mep_routing.routing import append_allowed_region_warning, count_segments_outside_allowed_region


def test_count_segments_outside_allowed_region_ignores_shaft_entry_segments():
    allowed_region = box(0, 0, 10, 10)
    shaft = Polygon([(10, 4), (12, 4), (12, 6), (10, 6)])
    routes = [
        ("Shaft", [((9, 5), (11, 5))]),
        ("Kitchen", [((1, 1), (9, 1)), ((9, 1), (12, 1))]),
    ]

    assert count_segments_outside_allowed_region(routes, allowed_region, shaft) == 1


def test_allowed_region_warning_handles_missing_region_and_outside_segments():
    routes = [("Bath", [((0, 0), (20, 0))])]

    assert append_allowed_region_warning(["existing"], routes, None) == ["existing"]
    assert append_allowed_region_warning(["existing"], routes, box(0, 0, 10, 10)) == [
        "existing",
        "1 segment(s) outside allowed",
    ]


def test_allowed_region_warning_handles_blocked_run_without_routes():
    assert count_segments_outside_allowed_region(None, box(0, 0, 10, 10)) == 0
    assert append_allowed_region_warning([], None, box(0, 0, 10, 10)) == []


def test_allowed_region_warning_accepts_installation_shaft_route_name():
    routes = [("Core", [((-2, 5), (2, 5))])]

    assert append_allowed_region_warning(
        [],
        routes,
        box(0, 0, 10, 10),
        box(-3, 4, 1, 6),
        shaft_route_name="Core",
    ) == []
