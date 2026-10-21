from __future__ import annotations

import math
import numpy as np
from mep_routing.geometry.directions import travel_direction, same_direction

from shapely.geometry import LineString, Point

from mep_routing.geometry import axis_segment_distance, axis_segment_relation
from mep_routing.routing.segments import (
    merged_route_axis_segments,
    metric_route_segments,
    point_is_segment_endpoint,
)


def count_segment_crossings(routes):
    crossing_points = set()
    all_segs = metric_route_segments(routes)
    if not all_segs:
        return 0
    lines = [LineString(segment) for _name, segment in all_segs]
    bounds = np.asarray([line.bounds for line in lines])
    names = np.asarray([name for name, _segment in all_segs])

    for i in range(len(all_segs)):
        name1, seg1 = all_segs[i]
        line1 = lines[i]
        # Skeleton polylines have many short diagonal edges. Reject separated
        # boxes numerically before invoking GEOS; preserve crossing semantics.
        candidates = np.flatnonzero(
            (np.arange(len(all_segs)) > i) & (names != name1)
            & (bounds[:, 0] <= bounds[i, 2]) & (bounds[:, 2] >= bounds[i, 0])
            & (bounds[:, 1] <= bounds[i, 3]) & (bounds[:, 3] >= bounds[i, 1])
        )
        for j in candidates:
            name2, seg2 = all_segs[j]
            line2 = lines[j]
            inter = line1.intersection(line2)
            if inter.is_empty or not isinstance(inter, Point):
                continue
            pt = (float(inter.x), float(inter.y))
            if point_is_segment_endpoint(pt, seg1) and point_is_segment_endpoint(pt, seg2):
                continue
            pair = tuple(sorted((name1, name2)))
            crossing_points.add((pair, round(pt[0], 6), round(pt[1], 6)))
    return len(crossing_points)


def count_segment_clearance_conflicts(routes, route_diameter, required_clearance):
    conflicts = 0
    all_segs = [
        (name, seg, route_diameter(name))
        for name, seg in merged_route_axis_segments(routes)
    ]

    for i, (name_a, seg_a, diameter_a) in enumerate(all_segs):
        for name_b, seg_b, diameter_b in all_segs[i + 1:]:
            if name_a == name_b:
                continue
            if axis_segment_relation(seg_a, seg_b) is not None:
                continue
            if axis_segment_distance(seg_a, seg_b) < required_clearance(diameter_a, diameter_b):
                conflicts += 1
    return conflicts


def count_segment_overlaps(routes):
    overlaps = 0
    all_segs = merged_route_axis_segments(routes)

    for i, (name_a, seg_a) in enumerate(all_segs):
        for name_b, seg_b in all_segs[i + 1:]:
            if name_a == name_b:
                continue
            if axis_segment_relation(seg_a, seg_b) == "overlap":
                overlaps += 1
    return overlaps


def segment_metric_dir(route_name, idx, p1, p2, eps=1e-7, *, include_diagonal=False):
    if include_diagonal:
        return travel_direction(p1, p2)
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    length = math.hypot(dx, dy)
    if length < eps:
        return None
    if abs(dy) < eps:
        return "E" if dx > 0 else "W"
    if abs(dx) < eps:
        return "N" if dy > 0 else "S"
    if route_name == "Shaft" and idx == 0:
        return (round(dx / length, 6), round(dy / length, 6))
    return None


def count_ordered_route_turns(route_name, segs, *, include_diagonal=False):
    """Count graph/pin turns; ignore diagonal snap artifacts except the shaft connector."""
    prev_dir = None
    turns = 0
    for idx, (p1, p2) in enumerate(segs):
        curr_dir = segment_metric_dir(route_name, idx, p1, p2, include_diagonal=include_diagonal)
        if curr_dir is None:
            continue

        if prev_dir is not None and not same_direction(curr_dir, prev_dir):
            turns += 1
        prev_dir = curr_dir
    return turns


def count_tree_route_turns(route_name, segs, *, include_diagonal=False):
    """Count directed branch turns in a rooted, flattened tree geometry."""
    adjacency = {}
    root = None
    for idx, (p1, p2) in enumerate(segs):
        direction = segment_metric_dir(route_name, idx, p1, p2, include_diagonal=include_diagonal)
        if direction is None or (not include_diagonal and not isinstance(direction, str)):
            continue
        left = (round(float(p1[0]), 6), round(float(p1[1]), 6))
        right = (round(float(p2[0]), 6), round(float(p2[1]), 6))
        if left == right:
            continue
        root = left if root is None else root
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    if root is None:
        return 0

    turns, visited = 0, {root}
    pending = [(root, None)]
    while pending:
        node, parent = pending.pop()
        incoming = None if parent is None else segment_metric_dir(
            route_name, 0, parent, node, include_diagonal=include_diagonal,
        )
        for child in adjacency.get(node, ()):
            if child in visited:
                continue
            visited.add(child)
            outgoing = segment_metric_dir(route_name, 0, node, child, include_diagonal=include_diagonal)
            if incoming is not None and outgoing is not None and not same_direction(incoming, outgoing):
                turns += 1
            pending.append((child, node))
    return turns


def count_solution_turns(routes, *, include_diagonal=False):
    return sum(
        count_tree_route_turns(name, segs, include_diagonal=include_diagonal)
        if name in ("Supply Air Tree", "Return Air Tree")
        else count_ordered_route_turns(name, segs, include_diagonal=include_diagonal)
        for name, segs in routes
    )


def merged_route_piece_lengths(route_name, segs, *, include_diagonal=False):
    if not segs:
        return []
    pieces = []
    current_dir = None
    current_len = 0.0
    for idx, (p1, p2) in enumerate(segs):
        length = math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))
        if length < 1.0:
            continue
        seg_dir = segment_metric_dir(route_name, idx, p1, p2, include_diagonal=include_diagonal)
        if seg_dir is None:
            if current_len > 0.0:
                pieces.append(current_len)
            pieces.append(length)
            current_dir = None
            current_len = 0.0
        elif same_direction(seg_dir, current_dir):
            current_len += length
        else:
            if current_len > 0.0:
                pieces.append(current_len)
            current_dir = seg_dir
            current_len = length
    if current_len > 0.0:
        pieces.append(current_len)
    return pieces


def count_route_short_pieces(route_name, segs, min_piece_length, *, include_diagonal=False):
    pieces = merged_route_piece_lengths(route_name, segs, include_diagonal=include_diagonal)
    if not pieces:
        return 0
    count = 0
    last_idx = len(pieces) - 1
    for idx, length in enumerate(pieces):
        min_len = min_piece_length(route_name, terminal_segment=(idx == 0 or idx == last_idx))
        if length + 1e-7 < min_len:
            count += 1
    return count


def count_solution_short_pieces(routes, min_piece_length, *, include_diagonal=False):
    return sum(count_route_short_pieces(name, segs, min_piece_length, include_diagonal=include_diagonal) for name, segs in routes)
