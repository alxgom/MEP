"""Placement-mode coordination independent of interactive application state."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlacementOutcome:
    position: tuple[float, float] | None
    rotation: float | None
    score: float | None
    scores: object
    fields: object
    candidate_count: int = 0


def run_core_like_placement(candidate_rooms, candidate_room_points, rotations, is_valid, score_candidate, choose_core):
    """Evaluate the core-like placement mode through injected domain callbacks."""
    selected, candidate_count = choose_core(
        candidate_rooms, candidate_room_points, rotations, is_valid, score_candidate,
    )
    if selected is None:
        return PlacementOutcome(None, None, None, {}, {}, candidate_count)
    x, y, rotation, score = selected
    return PlacementOutcome((x, y), rotation, score, {}, {}, candidate_count)


def run_topological_placement(env, shaft_nodes, rotations, is_valid, machine_pins, nearest_node, wet_room_names, weights, calculate_scores, choose_topological):
    """Evaluate topological fields and return a placement without mutating app state."""
    scores, fields = calculate_scores(env, shaft_nodes)
    if not scores:
        return PlacementOutcome(None, None, None, scores, fields)
    selected = choose_topological(
        env, scores, fields, rotations, is_valid, machine_pins, nearest_node,
        wet_room_names, weights,
    )
    if selected is None:
        return PlacementOutcome(None, None, None, scores, fields)
    x, y, rotation, score = selected
    return PlacementOutcome((x, y), rotation, score, scores, fields)
