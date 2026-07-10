"""Generic geometry helpers for the routing workbench."""

from .axis import axis_segment_distance, axis_segment_relation, normalize_axis_segment
from .distances import (
    edge_parallel_segment_min_distances,
    edge_segment_min_distances,
    point_segment_min_distances,
)
from .polygons import iter_polygons
from .rays import cast_rays_numpy, ray_ray_intersections_numpy
from .segments import extract_boundary_segments, extract_line_segments
from .shapely_utils import snap_to_integer_grid

__all__ = [
    "axis_segment_distance",
    "axis_segment_relation",
    "cast_rays_numpy",
    "edge_parallel_segment_min_distances",
    "edge_segment_min_distances",
    "extract_boundary_segments",
    "extract_line_segments",
    "iter_polygons",
    "normalize_axis_segment",
    "point_segment_min_distances",
    "ray_ray_intersections_numpy",
    "snap_to_integer_grid",
]
