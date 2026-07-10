import numpy as np

from vent_router.graphs import EnvView
from vent_router.placement import (
    choose_topological_machine_placement,
    pin_nodes_from_pins,
    rotation_score_from_fields,
)


def test_pin_nodes_from_pins_skips_corner_points():
    pins = {
        "left_mid": (0, 0),
        "right_mid": (1, 0),
        "c_tl": (0, 1),
    }

    assert pin_nodes_from_pins(pins, nearest_node_fn=lambda pt: pt[0] + 10) == {
        "left_mid": 10,
        "right_mid": 11,
    }


def test_rotation_score_from_fields_matches_current_large_and_small_pin_policy():
    pin_nodes = {
        "left_mid": 1,
        "right_mid": 2,
        "tl": 3,
        "tr": 4,
        "bl": 5,
        "br": 6,
    }
    distance_fields = {
        "Shaft": {1: 10.0, 2: 20.0},
        "Kitchen": {2: 5.0},
        "Bathroom": {3: 7.0, 4: 8.0, 5: 9.0, 6: 10.0},
        "Toilet": {3: 1.0, 4: 2.0, 5: 3.0, 6: 4.0},
    }

    score = rotation_score_from_fields(
        pin_nodes,
        distance_fields,
        wet_room_names=["Kitchen", "Bathroom", "Toilet"],
        weights={"Shaft": 2.0, "Kitchen": 3.0},
    )

    assert score == 44.0


def test_choose_topological_machine_placement_uses_score_order_and_first_valid_rotation():
    env = EnvView(
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        {0: [], 1: []},
    )
    node_scores = {0: 1.0, 1: 0.0}
    distance_fields = {
        "Shaft": {1: 10.0, 2: 20.0},
        "Kitchen": {2: 5.0},
    }
    pins = {
        "left_mid": "left",
        "right_mid": "right",
        "tl": "tl",
        "tr": "tr",
        "bl": "bl",
        "br": "br",
        "c_tl": "corner",
    }
    nearest = {
        "left": 1,
        "right": 2,
        "tl": 3,
        "tr": 4,
        "bl": 5,
        "br": 6,
    }

    selected = choose_topological_machine_placement(
        env,
        node_scores,
        distance_fields,
        rotations=(0, 90),
        is_valid_fn=lambda x, _y, rot: x == 10.0 and rot == 90,
        pins_fn=lambda _x, _y, _rot: pins,
        nearest_node_fn=lambda pt: nearest[pt],
        wet_room_names=["Kitchen"],
        weights={"Shaft": 2.0, "Kitchen": 3.0},
    )

    assert selected == (10.0, 0.0, 90, 35.0)
