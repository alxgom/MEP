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
    selected_pin_names,
)
from .clearance import (
    buffered_radius_mm,
    required_clearance_mm,
    route_axis_records,
    weighted_edge_cost,
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
    total_route_length_m,
)
from .search import (
    line_graph_dir_from_points,
    path_physical_length,
    terminal_node_indices,
    target_heuristic,
)

__all__ = [
    "RouteScoreWeights",
    "add_edge",
    "add_port_stub_segment",
    "buffered_radius_mm",
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
    "line_graph_dir_from_points",
    "path_physical_length",
    "point_is_segment_endpoint",
    "positive_flow_edges",
    "route_conflict_summary",
    "required_clearance_mm",
    "route_axis_records",
    "route_quality_counts",
    "route_quality_warnings",
    "score_routes",
    "selected_pin_names",
    "segment_metric_dir",
    "source_start_nodes",
    "target_heuristic",
    "terminal_node_indices",
    "total_route_length",
    "total_route_length_m",
    "trace_flow_path",
    "weighted_edge_cost",
]
