from __future__ import annotations

import math
from dataclasses import dataclass

from vent_router.routing.metrics import (
    count_segment_clearance_conflicts,
    count_segment_crossings,
    count_segment_overlaps,
    count_solution_short_pieces,
    count_solution_turns,
)


@dataclass(frozen=True)
class RouteScoreWeights:
    bend: float
    crossing: float
    overlap: float
    clearance: float
    short_piece: float


def total_route_length(routes):
    total_len = 0.0
    for _name, segs in routes:
        total_len += sum(
            math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))
            for p1, p2 in segs
        )
    return total_len


def total_route_length_m(routes):
    if not routes:
        return 0.0
    return total_route_length(routes) / 1000.0


def route_quality_counts(
    routes,
    route_diameter,
    required_clearance,
    min_piece_length,
    crossings=None,
):
    if crossings is None:
        crossings = count_segment_crossings(routes)
    return {
        "crossings": crossings,
        "turns": count_solution_turns(routes),
        "overlaps": count_segment_overlaps(routes),
        "clearance_conflicts": count_segment_clearance_conflicts(routes, route_diameter, required_clearance),
        "short_pieces": count_solution_short_pieces(routes, min_piece_length),
    }


def score_routes(
    routes,
    weights,
    route_diameter,
    required_clearance,
    min_piece_length,
    crossings=None,
):
    counts = route_quality_counts(
        routes,
        route_diameter,
        required_clearance,
        min_piece_length,
        crossings=crossings,
    )
    return (
        int(total_route_length(routes))
        + int(weights.bend) * counts["turns"]
        + int(weights.crossing) * counts["crossings"]
        + int(weights.overlap) * counts["overlaps"]
        + int(weights.clearance) * counts["clearance_conflicts"]
        + int(weights.short_piece) * counts["short_pieces"]
    )


def route_quality_warnings(routes, route_diameter, required_clearance, min_piece_length):
    if not routes:
        return []
    warnings = []
    counts = route_quality_counts(routes, route_diameter, required_clearance, min_piece_length)
    if counts["crossings"]:
        warnings.append(f'{counts["crossings"]} crossing(s)')
    if counts["overlaps"]:
        warnings.append(f'{counts["overlaps"]} overlap(s)')
    if counts["clearance_conflicts"]:
        warnings.append(f'{counts["clearance_conflicts"]} clearance conflict(s)')
    if counts["short_pieces"]:
        warnings.append(f'{counts["short_pieces"]} short piece(s)')
    return warnings


def route_conflict_summary(routes, route_diameter, required_clearance, min_piece_length):
    if not routes:
        return "no routes"
    counts = route_quality_counts(routes, route_diameter, required_clearance, min_piece_length)
    parts = [f'{counts["crossings"]} crossings']
    if counts["overlaps"]:
        parts.append(f'{counts["overlaps"]} overlaps')
    if counts["clearance_conflicts"]:
        parts.append(f'{counts["clearance_conflicts"]} clearance')
    return ", ".join(parts)
