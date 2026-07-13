from __future__ import annotations

import math


def normalize_axis_segment(p1, p2, eps=1e-7):
    """Normalize an axis-aligned segment to ordered coordinates plus direction."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    if abs(x1 - x2) < eps and abs(y1 - y2) < eps:
        return None
    if abs(y1 - y2) < eps:
        return (min(x1, x2), y1, max(x1, x2), y1, "H")
    if abs(x1 - x2) < eps:
        return (x1, min(y1, y2), x1, max(y1, y2), "V")
    return None


def axis_segment_relation(a, b, eps=1e-7):
    """Return overlap/cross relation for normalized axis-aligned segments."""
    ax1, ay1, ax2, ay2, a_dir = a
    bx1, by1, bx2, by2, b_dir = b

    if a_dir == b_dir:
        if a_dir == "H" and abs(ay1 - by1) < eps:
            return "overlap" if min(ax2, bx2) - max(ax1, bx1) > eps else None
        if a_dir == "V" and abs(ax1 - bx1) < eps:
            return "overlap" if min(ay2, by2) - max(ay1, by1) > eps else None
        return None

    h = a if a_dir == "H" else b
    v = a if a_dir == "V" else b
    hx1, hy, hx2, _, _ = h
    vx, vy1, _, vy2, _ = v
    if hx1 - eps <= vx <= hx2 + eps and vy1 - eps <= hy <= vy2 + eps:
        return "cross"
    return None


def axis_segment_distance(a, b):
    """Return Euclidean distance between normalized axis-aligned segment boxes."""
    ax1, ay1, ax2, ay2, _ = a
    bx1, by1, bx2, by2, _ = b
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return math.hypot(dx, dy)

