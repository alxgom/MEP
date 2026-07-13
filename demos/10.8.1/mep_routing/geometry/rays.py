from __future__ import annotations

import numpy as np


def cast_rays_numpy(interest_pts_arr, boundary_segments, eps=0.5):
    """Cast horizontal and vertical rays from interest points to boundary segments."""
    h_segs, v_segs = [], []
    dx_s = boundary_segments[:, 2] - boundary_segments[:, 0]
    dy_s = boundary_segments[:, 3] - boundary_segments[:, 1]

    for x0, y0 in interest_pts_arr:
        nh = np.abs(dy_s) > eps
        if np.any(nh):
            t_h = (y0 - boundary_segments[nh, 1]) / dy_s[nh]
            ok_h = (t_h >= -eps) & (t_h <= 1.0 + eps)
            x_i = boundary_segments[nh, 0] + t_h * dx_s[nh]

            east = x_i[ok_h & (x_i > x0 + eps)]
            if len(east):
                h_segs.append((y0, x0, float(east.min())))

            west = x_i[ok_h & (x_i < x0 - eps)]
            if len(west):
                h_segs.append((y0, float(west.max()), x0))

        nv = np.abs(dx_s) > eps
        if np.any(nv):
            t_v = (x0 - boundary_segments[nv, 0]) / dx_s[nv]
            ok_v = (t_v >= -eps) & (t_v <= 1.0 + eps)
            y_i = boundary_segments[nv, 1] + t_v * dy_s[nv]

            north = y_i[ok_v & (y_i > y0 + eps)]
            if len(north):
                v_segs.append((x0, y0, float(north.min())))

            south = y_i[ok_v & (y_i < y0 - eps)]
            if len(south):
                v_segs.append((x0, float(south.max()), y0))

    return h_segs, v_segs


def ray_ray_intersections_numpy(h_segs, v_segs, eps=0.5):
    """Return intersections between horizontal and vertical ray segments."""
    if not h_segs or not v_segs:
        return []
    h = np.array(h_segs, dtype=np.float64)
    v = np.array(v_segs, dtype=np.float64)

    y_h = h[:, 0:1]
    x1_h = h[:, 1:2]
    x2_h = h[:, 2:3]
    x_v = v[:, 0:1].T
    y1_v = v[:, 1:2].T
    y2_v = v[:, 2:3].T

    cross = (
        (x_v >= x1_h - eps)
        & (x_v <= x2_h + eps)
        & (y_h >= y1_v - eps)
        & (y_h <= y2_v + eps)
    )

    hi, vi = np.where(cross)
    return [(float(v[j, 0]), float(h[i, 0])) for i, j in zip(hi, vi)]

