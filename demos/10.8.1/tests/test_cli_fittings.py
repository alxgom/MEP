import pytest

from mep_routing.installations.cli.fittings import FittingCounts, count_tree_fittings, junction_fitting_count


@pytest.mark.parametrize("ends,expected", [
    (((-100, 0), (100, 0)), FittingCounts()),
    (((-100, 0), (0, 100)), FittingCounts(elbows=1)),
    (((-100, 0), (100, 0), (0, 100)), FittingCounts(tees=1)),
    (((-100, 0), (100, 0), (0, 100), (0, -100)), FittingCounts(crosses=1)),
])
def test_physical_ports_define_one_fitting(ends, expected):
    lines = [((0, 0), end) for end in ends]
    assert count_tree_fittings(lines) == expected
    assert count_tree_fittings([(end, start) for start, end in reversed(lines)]) == expected


def test_subdivisions_duplicate_rays_and_zero_aliases_do_not_add_fittings():
    lines = (((-100, 0), (-50, 0)), ((-50, 0), (0, 0)),
             ((0, 0), (0, 50)), ((0, 50), (0, 100)),
             ((0, 0), (0, 50)), ((0, 0), (0, 0)))
    assert count_tree_fittings(lines) == FittingCounts(elbows=1)


def test_local_terminal_takeoff_is_a_tee_but_leaf_connection_is_not_a_fitting():
    lines = (((-100, 0), (0, 0)), ((0, 0), (100, 0)))
    assert count_tree_fittings(lines, ((0, 0), (100, 0))) == FittingCounts(tees=1)


def test_graph_port_mask_classification_matches_physical_shapes():
    # Endpoint marker=1; E=2, W=4, N=8, S=16.
    straight = {2 | 4, 8 | 16}
    assert junction_fitting_count(2, 4, straight) == 0
    assert junction_fitting_count(8, 4, straight) == 1
    assert junction_fitting_count(2 | 8, 4, straight) == 1
    assert junction_fitting_count(2 | 8 | 16, 4, straight) == 1
    assert junction_fitting_count(2 | 1, 4, straight) == 1
    assert junction_fitting_count(1, 4, straight) == 0
    assert junction_fitting_count(2 | 4, 0, straight) == 0
