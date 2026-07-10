from __future__ import annotations

import math

from vent_router.geometry import normalize_axis_segment


def buffered_radius_mm(diameter_mm, buffer_ratio):
    return int(math.ceil(float(diameter_mm) / 2.0 * buffer_ratio))


def required_clearance_mm(diameter_a, diameter_b, buffer_ratio):
    return buffered_radius_mm(diameter_a, buffer_ratio) + buffered_radius_mm(diameter_b, buffer_ratio)


def route_axis_records(route_name, route_segs, route_diameter):
    diameter = route_diameter(route_name)
    records = []
    for p1, p2 in route_segs:
        seg = normalize_axis_segment(p1, p2)
        if seg is not None:
            records.append((seg, diameter))
    return records


def weighted_edge_cost(edge_weights, u, v, dist):
    if edge_weights is None:
        return dist
    return edge_weights.get((min(u, v), max(u, v)), dist)
