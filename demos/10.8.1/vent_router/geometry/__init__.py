"""Generic geometry helpers for the routing workbench."""

from .distances import (
    edge_parallel_segment_min_distances,
    edge_segment_min_distances,
    point_segment_min_distances,
)
from .shapely_utils import snap_to_integer_grid

__all__ = [
    "edge_parallel_segment_min_distances",
    "edge_segment_min_distances",
    "point_segment_min_distances",
    "snap_to_integer_grid",
]
