"""Generic geometry helpers for the routing workbench."""

from .distances import (
    edge_parallel_segment_min_distances,
    edge_segment_min_distances,
    point_segment_min_distances,
)
from .rays import cast_rays_numpy, ray_ray_intersections_numpy
from .segments import extract_boundary_segments, extract_line_segments
from .shapely_utils import snap_to_integer_grid

__all__ = [
    "cast_rays_numpy",
    "edge_parallel_segment_min_distances",
    "edge_segment_min_distances",
    "extract_boundary_segments",
    "extract_line_segments",
    "point_segment_min_distances",
    "ray_ray_intersections_numpy",
    "snap_to_integer_grid",
]
