from vent_router.routing import find_route_at_point, find_route_hit_at_point


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
