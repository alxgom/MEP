"""Routing helpers and algorithms for the interactive workbench."""

from .segments import (
    merged_axis_segments,
    merged_route_axis_segments,
    metric_route_segments,
    point_is_segment_endpoint,
)
from .metrics import (
    count_ordered_route_turns,
    count_route_short_pieces,
    count_segment_clearance_conflicts,
    count_segment_crossings,
    count_segment_overlaps,
    count_solution_short_pieces,
    count_solution_turns,
    merged_route_piece_lengths,
    segment_metric_dir,
)
from .scoring import (
    RouteScoreWeights,
    route_conflict_summary,
    route_quality_counts,
    route_quality_warnings,
    score_routes,
    total_route_length,
)

__all__ = [
    "RouteScoreWeights",
    "count_ordered_route_turns",
    "count_route_short_pieces",
    "count_segment_clearance_conflicts",
    "count_segment_crossings",
    "count_segment_overlaps",
    "count_solution_short_pieces",
    "count_solution_turns",
    "merged_axis_segments",
    "merged_route_axis_segments",
    "merged_route_piece_lengths",
    "metric_route_segments",
    "point_is_segment_endpoint",
    "route_conflict_summary",
    "route_quality_counts",
    "route_quality_warnings",
    "score_routes",
    "segment_metric_dir",
    "total_route_length",
]
