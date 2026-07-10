"""Routing helpers and algorithms for the interactive workbench."""

from .segments import (
    add_port_stub_segment,
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
from .hit_testing import (
    find_route_at_point,
    find_route_hit_at_point,
)
from .flow import (
    add_edge,
    min_cost_flow,
    positive_flow_edges,
    source_start_nodes,
    trace_flow_path,
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
    "add_edge",
    "add_port_stub_segment",
    "count_ordered_route_turns",
    "count_route_short_pieces",
    "count_segment_clearance_conflicts",
    "count_segment_crossings",
    "count_segment_overlaps",
    "count_solution_short_pieces",
    "count_solution_turns",
    "find_route_at_point",
    "find_route_hit_at_point",
    "merged_axis_segments",
    "merged_route_axis_segments",
    "merged_route_piece_lengths",
    "metric_route_segments",
    "min_cost_flow",
    "point_is_segment_endpoint",
    "positive_flow_edges",
    "route_conflict_summary",
    "route_quality_counts",
    "route_quality_warnings",
    "score_routes",
    "segment_metric_dir",
    "source_start_nodes",
    "total_route_length",
    "trace_flow_path",
]
