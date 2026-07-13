from mep_routing.routing import (
    RouteScoreWeights,
    route_conflict_summary,
    route_quality_warnings,
    score_routes,
    total_route_length,
    total_route_length_m,
)


def _diameter(_route_name):
    return 10


def _clearance(_diameter_a, _diameter_b):
    return 8


def _min_piece(_route_name, terminal_segment=False):
    return 12 if terminal_segment else 24


def test_total_route_length_sums_segment_lengths():
    routes = [
        ("A", [((0, 0), (3, 4))]),
        ("B", [((0, 0), (0, 10))]),
    ]

    assert total_route_length(routes) == 15.0


def test_total_route_length_m_converts_from_millimetres():
    routes = [("A", [((0, 0), (3000, 4000))])]

    assert total_route_length_m(routes) == 5.0
    assert total_route_length_m([]) == 0.0


def test_score_routes_combines_length_and_quality_penalties():
    routes = [
        ("A", [((0, 0), (10, 0)), ((10, 0), (10, 10))]),
        ("B", [((5, -5), (5, 5))]),
    ]
    weights = RouteScoreWeights(
        bend=100,
        crossing=1000,
        overlap=2000,
        clearance=3000,
        short_piece=4000,
    )

    assert score_routes(routes, weights, _diameter, _clearance, _min_piece) == 16130


def test_route_quality_warnings_formats_only_present_issues():
    routes = [
        ("A", [((0, 0), (10, 0))]),
        ("B", [((0, 5), (10, 5))]),
    ]

    assert route_quality_warnings(routes, _diameter, _clearance, _min_piece) == [
        "1 clearance conflict(s)",
        "2 short piece(s)",
    ]


def test_route_conflict_summary_keeps_baseline_crossing_text():
    routes = [
        ("A", [((0, 0), (10, 0))]),
        ("B", [((0, 5), (10, 5))]),
    ]

    assert route_conflict_summary(routes, _diameter, _clearance, _min_piece) == (
        "0 crossings, 1 clearance"
    )
