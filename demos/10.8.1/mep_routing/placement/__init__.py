"""Machine placement helpers for the interactive routing workbench."""

from .application import PlacementApplicationAdapter
from .feasibility import candidate_machine_rooms, is_machine_placement_valid
from .vertical_clearance import (
    available_machine_placement_region,
    insufficient_machine_clearance_regions,
    scores_outside_regions,
)
from .fields import compute_dijkstra_distance_field, placement_weights, topological_placement_scores
from .rotation import (
    field_alignment_pin_dirs,
    rotation_field_rooms_for_pin,
    rotation_room_weight,
    score_rotation_field_at,
    select_field_alignment_rotation,
)
from .selection import (
    best_valid_rotation_for_point,
    choose_core_like_machine_placement,
    choose_topological_machine_placement,
    pin_nodes_from_pins,
    rotation_score_from_fields,
)
from .scoring import (
    area_out_percentage,
    candidate_room_points,
    core_like_machine_candidate_score,
    machine_polygon_from_pins,
    point_angle_to_target,
    routing_frame_axes,
)

__all__ = [
    "PlacementApplicationAdapter",
    "area_out_percentage",
    "available_machine_placement_region",
    "best_valid_rotation_for_point",
    "candidate_machine_rooms",
    "candidate_room_points",
    "choose_core_like_machine_placement",
    "choose_topological_machine_placement",
    "compute_dijkstra_distance_field",
    "core_like_machine_candidate_score",
    "field_alignment_pin_dirs",
    "machine_polygon_from_pins",
    "placement_weights",
    "pin_nodes_from_pins",
    "point_angle_to_target",
    "is_machine_placement_valid",
    "insufficient_machine_clearance_regions",
    "scores_outside_regions",
    "rotation_field_rooms_for_pin",
    "rotation_room_weight",
    "rotation_score_from_fields",
    "routing_frame_axes",
    "score_rotation_field_at",
    "select_field_alignment_rotation",
    "topological_placement_scores",
]
