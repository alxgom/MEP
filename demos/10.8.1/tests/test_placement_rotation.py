import pytest

from mep_routing.placement import (
    field_alignment_pin_dirs,
    rotation_field_rooms_for_pin,
    rotation_room_weight,
    score_rotation_field_at,
    select_field_alignment_rotation,
)


def test_rotation_field_rooms_for_pin_routes_large_and_small_pins():
    assert rotation_field_rooms_for_pin("left_mid", ["Bathroom", "Kitchen"], {"Kitchen"}) == ["Shaft", "Kitchen"]
    assert rotation_field_rooms_for_pin("tl", ["Bathroom", "Kitchen", "Toilet"], {"Kitchen"}) == ["Bathroom", "Toilet"]


def test_rotation_room_weight_matches_current_modes():
    assert rotation_room_weight("Shaft", weight_mode_idx=0) == 2.0
    assert rotation_room_weight("Kitchen", weight_mode_idx=0) == 1.5
    assert rotation_room_weight("Bathroom", weight_mode_idx=0) == 1.0
    assert rotation_room_weight("Shaft", weight_mode_idx=1) == 1.0


def test_field_alignment_pin_dirs_uses_supplied_axis_transform():
    dirs = field_alignment_pin_dirs(
        "tl",
        90,
        local_axis_to_world_fn=lambda local_vec, angle: (local_vec[0], local_vec[1], angle),
    )

    assert dirs == [(-1.0, 0.0, 90), (0.0, 1.0, 90)]


def test_score_rotation_field_at_rewards_aligned_large_pin_to_shaft():
    pins = {
        "left_mid": (-1, 0),
        "right_mid": (1, 0),
        "tl": (-1, 1),
        "tr": (1, 1),
        "bl": (-1, -1),
        "br": (1, -1),
    }

    score = score_rotation_field_at(
        pins,
        angle=0,
        wet_room_names=[],
        terminal_names=set(),
        shaft_point=(-251, 0),
        room_target_fn=lambda _name: None,
        weight_mode_idx=0,
        local_axis_to_world_fn=lambda local_vec, _angle: local_vec,
    )

    assert score == pytest.approx(2.0 / 250.0)


def test_select_field_alignment_rotation_stays_when_current_orientation_wins():
    new_angle, selected, scores = select_field_alignment_rotation(
        current_angle=0,
        is_valid_rotation_fn=lambda _angle: True,
        score_rotation_fn=lambda angle: {0: 10.0, 180: 5.0, 90: 8.0, 270: 7.0}[angle],
        eps=0.01,
    )

    assert new_angle == 0
    assert selected == "H"
    assert scores == {"H": 10.0, "V": 8.0}


def test_select_field_alignment_rotation_switches_when_other_orientation_exceeds_epsilon():
    new_angle, selected, scores = select_field_alignment_rotation(
        current_angle=0,
        is_valid_rotation_fn=lambda angle: angle != 180,
        score_rotation_fn=lambda angle: {0: 10.0, 90: 10.2, 270: 9.0}[angle],
        eps=0.1,
    )

    assert new_angle == 90
    assert selected == "V"
    assert scores == {"H": 10.0, "V": 10.2}
