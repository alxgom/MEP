"""Generic geometry helpers for the routing workbench."""

from .distances import (
    edge_parallel_segment_min_distances,
    edge_segment_min_distances,
    point_segment_min_distances,
)
from .segments import extract_boundary_segments, extract_line_segments
from .shapely_utils import snap_to_integer_grid

__all__ = [
    "edge_parallel_segment_min_distances",
    "edge_segment_min_distances",
    "extract_boundary_segments",
    "extract_line_segments",
    "point_segment_min_distances",
    "snap_to_integer_grid",
]
