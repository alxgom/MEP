"""Local wall-offset samples and validated orthogonal connections."""

import math

import numpy as np
from shapely.geometry import LineString

from mep_routing.geometry.segments import extract_line_segments
from .orthogonal import OrthogonalGrid


def build_shifted_wall_grid(walls, allowed_region, wall_polys=(), wall_thickness_mm=150,
                            *, offset_mm=100, sample_spacing_mm=200):
    """Sample both normal offsets, clipping around holes before making paths.

    Diagonal samples try both local L shapes. Their orthogonal steps must each
    pass the same geometry/wall checks as the cell skeleton. No full dwelling
    axes or unchecked diagonal edges are generated.
    """
    if not math.isfinite(offset_mm) or offset_mm < 0:
        raise ValueError("wall-grid offset must be finite and nonnegative")
    if not math.isfinite(sample_spacing_mm) or sample_spacing_mm <= 0:
        raise ValueError("wall-grid spacing must be finite and positive")
    grid = OrthogonalGrid(allowed_region, wall_polys, wall_thickness_mm)
    if allowed_region is None or allowed_region.is_empty:
        return grid.finish()
    for wall in walls:
        for x0, y0, x1, y1 in extract_line_segments(wall):
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length < 1e-4:
                continue
            normal = np.asarray((-dy / length, dx / length))
            for sign in (-1, 1) if offset_mm else (1,):
                shift = sign * offset_mm * normal
                line = LineString((np.asarray((x0, y0)) + shift, np.asarray((x1, y1)) + shift))
                for a, b, c, d in extract_line_segments(line.intersection(allowed_region)):
                    count = max(1, math.ceil(math.hypot(c - a, d - b) / sample_spacing_mm))
                    points = [tuple(round(float(x), 6) for x in point)
                              for point in np.linspace((a, b), (c, d), count + 1)]
                    for first, second in zip(points, points[1:]):
                        grid.add_path((first, (second[0], first[1]), second))
                        grid.add_path((first, (first[0], second[1]), second))
    return grid.finish()
