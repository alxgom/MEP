import numpy as np

from mep_routing.ui.terminal_selection import (
    apply_preferred_terminal_area,
    apply_preferred_terminal_point,
    map_preferred_points_to_nodes,
)


NODES = np.array(((0, 0), (100, 0), (200, 0), (0, 100)), dtype=np.float32)


def test_preferred_points_map_to_unique_nearby_candidate_nodes():
    mapped, indices = map_preferred_points_to_nodes([(1, 1), (3, 2), (198, 0)], [0, 1, 2], NODES, 10)
    assert mapped == [0, 2]
    assert indices == {0: 0, 2: 2}


def test_point_selection_adds_then_removes_nearest_candidate_preference():
    preferences = {}
    candidates = {"bath": [0, 1]}
    added = apply_preferred_terminal_point(preferences, (98, 0), False, ["bath"], lambda _room, _point: True, candidates.get, NODES, 20)
    removed = apply_preferred_terminal_point(preferences, (98, 0), True, ["bath"], lambda _room, _point: True, candidates.get, NODES, 20)
    assert added == (True, "bath")
    assert removed == (True, "bath")
    assert preferences == {}


def test_area_selection_tracks_added_nodes_and_removes_overlapping_area_metadata():
    preferences, areas = {}, []
    candidates = {"bath": [0, 1, 2], "wc": [3]}
    added = apply_preferred_terminal_area(preferences, areas, (-1, -1), (110, 10), False, candidates.keys(), candidates.get, NODES, 20)
    removed = apply_preferred_terminal_area(preferences, areas, (-1, -1), (110, 10), True, candidates.keys(), candidates.get, NODES, 20)
    assert added == (True, "bath")
    assert preferences == {}
    assert areas == []
    assert removed == (True, "bath")
