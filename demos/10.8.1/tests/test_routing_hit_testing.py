from vent_router.routing import find_route_at_point, find_route_hit_at_point, selected_pin_names


def test_find_route_hit_at_point_returns_nearest_route_with_distance():
    routes = [
        ("A", [((0, 0), (10, 0))]),
        ("B", [((0, 20), (10, 20))]),
    ]

    assert find_route_hit_at_point(routes, (5, 18), hit_radius_mm=5) == ("B", 2.0)


def test_find_route_at_point_returns_none_outside_hit_radius():
    routes = [("A", [((0, 0), (10, 0))])]

    assert find_route_at_point(routes, (5, 20), hit_radius_mm=5) is None


def test_find_route_at_point_returns_route_name_inside_hit_radius():
    routes = [("A", [((0, 0), (10, 0))])]

    assert find_route_at_point(routes, (5, 3), hit_radius_mm=5) == "A"


def test_selected_pin_names_matches_route_endpoint_pins():
    routes = [
        ("Bath", [((100, 100), (110, 100)), ((110, 100), (120, 100))]),
        ("Kitchen", [((0, 0), (10, 0)), ((10, 0), (20, 0)), ((20, 0), (30, 0))]),
    ]
    pins = {"left_mid": (0, 0), "right_mid": (30, 0), "tl": (999, 999)}

    assert selected_pin_names("Kitchen", routes, pins) == {"left_mid", "right_mid"}


def test_selected_pin_names_checks_only_last_three_segments():
    routes = [
        (
            "Kitchen",
            [
                ((0, 0), (10, 0)),
                ((10, 0), (20, 0)),
                ((20, 0), (30, 0)),
                ((30, 0), (40, 0)),
            ],
        )
    ]
    pins = {"left_mid": (0, 0), "right_mid": (40, 0)}

    assert selected_pin_names("Kitchen", routes, pins) == {"right_mid"}


def test_selected_pin_names_returns_empty_for_missing_context():
    assert selected_pin_names(None, [], {}) == set()
    assert selected_pin_names("Kitchen", [], {"left_mid": (0, 0)}) == set()
