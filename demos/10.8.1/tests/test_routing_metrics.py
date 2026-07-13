from mep_routing.routing import (
    count_route_short_pieces,
    count_segment_clearance_conflicts,
    count_segment_crossings,
    count_segment_overlaps,
    count_solution_turns,
    merged_route_piece_lengths,
)


def test_count_segment_crossings_ignores_shared_endpoints():
    routes = [
        ("A", [((0, 0), (10, 0))]),
        ("B", [((5, -5), (5, 5))]),
        ("C", [((10, 0), (20, 0))]),
    ]

    assert count_segment_crossings(routes) == 1


def test_count_segment_overlaps_counts_inter_route_axis_overlap():
    routes = [
        ("A", [((0, 0), (20, 0))]),
        ("B", [((10, 0), (30, 0))]),
        ("A", [((50, 0), (60, 0))]),
    ]

    assert count_segment_overlaps(routes) == 1


def test_count_segment_clearance_conflicts_uses_injected_policy():
    routes = [
        ("A", [((0, 0), (10, 0))]),
        ("B", [((0, 5), (10, 5))]),
    ]

    assert count_segment_clearance_conflicts(
        routes,
        route_diameter=lambda _name: 10,
        required_clearance=lambda _diameter_a, _diameter_b: 8,
    ) == 1


def test_count_solution_turns_ignores_non_shaft_diagonal_artifacts():
    routes = [
        ("A", [((0, 0), (10, 0)), ((10, 0), (20, 10)), ((20, 10), (20, 20))]),
        ("Shaft", [((0, 0), (3, 4)), ((3, 4), (3, 10))]),
    ]

    assert count_solution_turns(routes) == 2


def test_route_short_pieces_uses_injected_min_piece_length():
    segs = [((0, 0), (10, 0)), ((10, 0), (20, 0)), ((20, 0), (20, 25))]

    assert merged_route_piece_lengths("A", segs) == [20.0, 25.0]
    assert count_route_short_pieces(
        "A",
        segs,
        min_piece_length=lambda _route, terminal_segment=False: 24 if terminal_segment else 40,
    ) == 1
