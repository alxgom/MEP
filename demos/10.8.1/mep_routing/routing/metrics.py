from __future__ import annotations

import math

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

    for i in range(len(all_segs)):
        name1, seg1 = all_segs[i]
        line1 = LineString(seg1)

        for j in range(i + 1, len(all_segs)):
            name2, seg2 = all_segs[j]
            if name1 == name2:
                continue
            line2 = LineString(seg2)
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


def segment_metric_dir(route_name, idx, p1, p2, eps=1e-7):
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


def count_ordered_route_turns(route_name, segs):
    """Count graph/pin turns; ignore diagonal snap artifacts except the shaft connector."""
    prev_dir = None
    turns = 0
    for idx, (p1, p2) in enumerate(segs):
        curr_dir = segment_metric_dir(route_name, idx, p1, p2)
        if curr_dir is None:
            continue

        if prev_dir is not None and curr_dir != prev_dir:
            turns += 1
        prev_dir = curr_dir
    return turns


def count_solution_turns(routes):
    return sum(count_ordered_route_turns(name, segs) for name, segs in routes)


def merged_route_piece_lengths(route_name, segs):
    if not segs:
        return []
    pieces = []
    current_dir = None
    current_len = 0.0
    for idx, (p1, p2) in enumerate(segs):
        length = math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))
        if length < 1.0:
            continue
        seg_dir = segment_metric_dir(route_name, idx, p1, p2)
        if seg_dir is None:
            if current_len > 0.0:
                pieces.append(current_len)
            pieces.append(length)
            current_dir = None
            current_len = 0.0
        elif seg_dir == current_dir:
            current_len += length
        else:
            if current_len > 0.0:
                pieces.append(current_len)
            current_dir = seg_dir
            current_len = length
    if current_len > 0.0:
        pieces.append(current_len)
    return pieces


def count_route_short_pieces(route_name, segs, min_piece_length):
    pieces = merged_route_piece_lengths(route_name, segs)
    if not pieces:
        return 0
    count = 0
    last_idx = len(pieces) - 1
    for idx, length in enumerate(pieces):
        min_len = min_piece_length(route_name, terminal_segment=(idx == 0 or idx == last_idx))
        if length + 1e-7 < min_len:
            count += 1
    return count


def count_solution_short_pieces(routes, min_piece_length):
    return sum(count_route_short_pieces(name, segs, min_piece_length) for name, segs in routes)
