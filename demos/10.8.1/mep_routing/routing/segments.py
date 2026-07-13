from __future__ import annotations

import math

from mep_routing.geometry import normalize_axis_segment


def merged_axis_segments(route_segs, eps=1e-7):
    """Merge collinear axis-aligned route pieces by shared line."""
    by_line = {}
    for p1, p2 in route_segs:
        seg = normalize_axis_segment(p1, p2, eps=eps)
        if seg is None:
            continue
        x1, y1, x2, y2, axis = seg
        key = (axis, round(y1 if axis == "H" else x1, 6))
        interval = (x1, x2) if axis == "H" else (y1, y2)
        by_line.setdefault(key, []).append(interval)

    merged = []
    for (axis, coord), intervals in by_line.items():
        intervals.sort()
        start, end = intervals[0]
        for curr_start, curr_end in intervals[1:]:
            if curr_start <= end + eps:
                end = max(end, curr_end)
            else:
                if axis == "H":
                    merged.append((start, coord, end, coord, "H"))
                else:
                    merged.append((coord, start, coord, end, "V"))
                start, end = curr_start, curr_end
        if axis == "H":
            merged.append((start, coord, end, coord, "H"))
        else:
            merged.append((coord, start, coord, end, "V"))
    return merged


def merged_route_axis_segments(routes):
    """Return merged axis-aligned route segments with route names."""
    return [
        (name, seg)
        for name, route_segs in routes
        for seg in merged_axis_segments(route_segs)
    ]


def metric_route_segments(routes):
    """Return route segments for geometric metrics, preserving non-axis pieces."""
    segments = [
        (name, ((seg[0], seg[1]), (seg[2], seg[3])))
        for name, seg in merged_route_axis_segments(routes)
    ]
    for name, route_segs in routes:
        for p1, p2 in route_segs:
            if normalize_axis_segment(p1, p2) is not None:
                continue
            if math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1])) < 1e-7:
                continue
            segments.append((name, ((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))))
    return segments


def point_is_segment_endpoint(pt, seg, eps=1e-7):
    """Return whether a point matches either endpoint of a segment."""
    return (
        math.hypot(pt[0] - seg[0][0], pt[1] - seg[0][1]) < eps
        or math.hypot(pt[0] - seg[1][0], pt[1] - seg[1][1]) < eps
    )


def add_port_stub_segment(segs, pin_name, target_node_idx, global_pins, nodes, target_spec=None):
    if target_node_idx is None or pin_name not in global_pins:
        return
    node_pt = nodes[target_node_idx]
    access_pt = target_spec["access_point"] if target_spec else node_pt
    pin_pt = target_spec["pin_point"] if target_spec else global_pins[pin_name]

    node_pt = (float(node_pt[0]), float(node_pt[1]))
    access_pt = (float(access_pt[0]), float(access_pt[1]))
    pin_pt = (float(pin_pt[0]), float(pin_pt[1]))
    if math.hypot(access_pt[0] - node_pt[0], access_pt[1] - node_pt[1]) > 1e-7:
        segs.append((node_pt, access_pt))
    segs.append((access_pt, pin_pt))


def route_segments_from_path(
    route_name,
    path,
    nodes,
    shaft_entry_segments_fn=None,
    pin_name=None,
    global_pins=None,
    target=None,
):
    segs = []
    if route_name == "Shaft" and path and shaft_entry_segments_fn is not None:
        shaft_entry_segments_fn(segs, path[0])
    for i in range(len(path) - 1):
        p1 = nodes[path[i]]
        p2 = nodes[path[i + 1]]
        segs.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))
    if pin_name and global_pins is not None:
        add_port_stub_segment(segs, pin_name, path[-1], global_pins, nodes, target)
    return segs


def build_routes_from_paths(route_order, paths, targets, global_pins, route_segments_fn):
    routes = []
    total_nodes = 0
    for route_name in route_order:
        path = paths.get(route_name)
        target = targets.get(route_name)
        if path is None or target is None:
            return None, 0
        segs = route_segments_fn(route_name, path, target["pin"], global_pins, target)
        routes.append((route_name, segs))
        total_nodes += len(path)
    return routes, total_nodes
