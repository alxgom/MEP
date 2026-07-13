from __future__ import annotations

import math

import numpy as np

from mep_routing.geometry import (
    edge_parallel_segment_min_distances,
    edge_segment_min_distances,
    extract_boundary_segments,
    extract_line_segments,
    normalize_axis_segment,
)


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


def normalized_edge(u, v):
    return (min(int(u), int(v)), max(int(u), int(v)))


def set_block_weight(edge_weights, u, v, block_weight):
    edge = normalized_edge(u, v)
    edge_weights[edge] = block_weight
    return edge


def block_terminal_node_edges(edge_weights, adj, terminal_node_indices, block_weight, skip_names=("Shaft",)):
    blocked_edges = []
    for route_name, node_idx in terminal_node_indices.items():
        if route_name in skip_names:
            continue
        if node_idx not in adj:
            continue
        for nbr, _, _ in adj[node_idx]:
            blocked_edges.append(set_block_weight(edge_weights, node_idx, nbr, block_weight))
    return blocked_edges


def machine_edge_clearance_distances(
    edge_coords,
    machine_center,
    machine_angle_deg,
    machine_overall_width_mm,
    machine_body_height_mm,
):
    """Return each graph edge's sampled distance from the machine envelope."""
    coords = np.asarray(edge_coords, dtype=np.float64)
    if len(coords) == 0:
        return np.empty((0,), dtype=np.float64)

    samples_t = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    xs = coords[:, 0:1] + (coords[:, 2:3] - coords[:, 0:1]) * samples_t
    ys = coords[:, 1:2] + (coords[:, 3:4] - coords[:, 1:2]) * samples_t

    rad = math.radians(machine_angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    center_x, center_y = machine_center
    dx = xs - float(center_x)
    dy = ys - float(center_y)

    local_x = dx * cos_a + dy * sin_a
    local_y = -dx * sin_a + dy * cos_a
    outside_x = np.maximum(np.abs(local_x) - float(machine_overall_width_mm) / 2.0, 0.0)
    outside_y = np.maximum(np.abs(local_y) - float(machine_body_height_mm) / 2.0, 0.0)
    return np.min(np.hypot(outside_x, outside_y), axis=1)


def static_clearance_distances(edge_coords, wall_segments, shaft_segments):
    """Return graph-edge distances to static wall and shaft segments."""
    if edge_coords is None or len(edge_coords) == 0:
        return None, None
    return (
        edge_parallel_segment_min_distances(edge_coords, wall_segments),
        edge_segment_min_distances(edge_coords, shaft_segments),
    )


def static_wall_distance_segments(routing_region, room_polygons, walls, wall_polygons):
    """Build static wall-like constraints for graph-edge clearance distances."""
    segments = []

    def append_boundary(geometry):
        boundary = extract_boundary_segments(geometry)
        if len(boundary):
            segments.append(boundary)

    if routing_region is not None and not routing_region.is_empty:
        append_boundary(routing_region)
    for polygon in room_polygons:
        append_boundary(polygon)
    for wall in walls:
        line_segments = extract_line_segments(wall)
        if len(line_segments):
            segments.append(line_segments)
    for polygon in wall_polygons:
        append_boundary(polygon)
    return np.vstack(segments) if segments else np.empty((0, 4), dtype=np.float64)


def static_shaft_distance_segments(shafts):
    """Build static shaft boundary constraints for graph-edge clearance distances."""
    segments = [extract_boundary_segments(shaft) for shaft in shafts]
    segments = [segment_set for segment_set in segments if len(segment_set)]
    return np.vstack(segments) if segments else np.empty((0, 4), dtype=np.float64)


def static_clearance_cache_key(routing_region, grid_edge_list, room_polygons, wall_polygons, shafts):
    """Return a geometry-sensitive cache key for static graph clearance fields."""
    return (
        id(routing_region),
        len(grid_edge_list or []),
        len(room_polygons),
        len(wall_polygons),
        len(shafts),
        tuple(polygon.bounds for polygon in room_polygons),
        tuple(polygon.bounds for polygon in shafts),
        tuple(polygon.bounds for polygon in wall_polygons),
    )


def add_static_clearance_weights(
    edge_weights,
    edge_list,
    wall_distances,
    shaft_distances,
    route_diameter_mm,
    buffer_ratio,
    shaft_clearance_mm,
    block_weight,
    allow_shaft_entry=False,
):
    """Block graph edges that violate static wall or shaft clearance."""
    if wall_distances is None and shaft_distances is None:
        return

    radius = float(buffered_radius_mm(route_diameter_mm, buffer_ratio))
    blocked_mask = np.zeros(len(edge_list), dtype=bool)
    if wall_distances is not None:
        blocked_mask |= np.asarray(wall_distances) < radius - 1e-7
    if shaft_distances is not None and not allow_shaft_entry:
        shaft_limit = max(float(shaft_clearance_mm), radius)
        blocked_mask |= np.asarray(shaft_distances) < shaft_limit - 1e-7

    for edge_index in np.flatnonzero(blocked_mask):
        u, v, _, _ = edge_list[int(edge_index)]
        edge_weights[normalized_edge(u, v)] = block_weight


def add_machine_clearance_weights(
    edge_weights,
    edge_list,
    edge_coords,
    nodes,
    route_diameter_mm,
    buffer_ratio,
    machine_center,
    machine_angle_deg,
    machine_overall_width_mm,
    machine_body_height_mm,
    soft_margin_mm,
    clearance_penalty,
    block_weight,
):
    """Block or penalize graph edges near the active machine envelope."""
    if edge_coords is None or edge_list is None:
        return
    distances = machine_edge_clearance_distances(
        edge_coords,
        machine_center,
        machine_angle_deg,
        machine_overall_width_mm,
        machine_body_height_mm,
    )
    if len(distances) == 0:
        return

    radius = float(buffered_radius_mm(route_diameter_mm, buffer_ratio))
    soft_limit = radius + float(soft_margin_mm)
    hard_mask = distances < radius - 1e-7
    soft_mask = (distances >= radius - 1e-7) & (distances < soft_limit)

    for edge_index in np.flatnonzero(hard_mask):
        u, v, _, _ = edge_list[int(edge_index)]
        edge_weights[normalized_edge(u, v)] = block_weight

    soft_indices = np.flatnonzero(soft_mask)
    if len(soft_indices) == 0:
        return
    t = (soft_limit - distances[soft_indices]) / float(soft_margin_mm)
    penalties = float(clearance_penalty) * np.square(t)
    for edge_index, penalty in zip(soft_indices, penalties):
        u, v, _, _ = edge_list[int(edge_index)]
        edge = normalized_edge(u, v)
        if edge_weights.get(edge, 0.0) >= block_weight:
            continue
        base_dist = float(np.hypot(nodes[v][0] - nodes[u][0], nodes[v][1] - nodes[u][1]))
        edge_weights[edge] = edge_weights.get(edge, base_dist) + float(penalty)


def add_route_interaction_weights(
    prior_axis_records,
    current_diameter_mm,
    accumulated_weights,
    edge_list,
    edge_coords,
    nodes,
    buffer_ratio,
    crossing_penalty,
    clearance_penalty,
    block_weight,
):
    """Apply overlap blocks and crossing/clearance penalties from prior routes."""
    if not prior_axis_records or edge_coords is None or edge_list is None:
        return

    coords = np.asarray(edge_coords, dtype=np.float64)
    edge_x1 = np.minimum(coords[:, 0], coords[:, 2])
    edge_x2 = np.maximum(coords[:, 0], coords[:, 2])
    edge_y1 = np.minimum(coords[:, 1], coords[:, 3])
    edge_y2 = np.maximum(coords[:, 1], coords[:, 3])
    edge_is_h = np.abs(coords[:, 1] - coords[:, 3]) < 1e-7

    blocked_edges = set()
    crossing_counts = {}
    clearance_counts = {}

    for prior_seg, prior_diameter in prior_axis_records:
        px1, py1, px2, py2, prior_dir = prior_seg
        required = required_clearance_mm(current_diameter_mm, prior_diameter, buffer_ratio)

        if prior_dir == "H":
            overlap_mask = (
                edge_is_h
                & (np.abs(edge_y1 - py1) < 1e-7)
                & (np.minimum(edge_x2, px2) - np.maximum(edge_x1, px1) > 1e-7)
            )
            cross_mask = (
                ~edge_is_h
                & (edge_x1 >= px1 - 1e-7)
                & (edge_x1 <= px2 + 1e-7)
                & (edge_y1 <= py1 + 1e-7)
                & (edge_y2 >= py1 - 1e-7)
            )
        else:
            overlap_mask = (
                ~edge_is_h
                & (np.abs(edge_x1 - px1) < 1e-7)
                & (np.minimum(edge_y2, py2) - np.maximum(edge_y1, py1) > 1e-7)
            )
            cross_mask = (
                edge_is_h
                & (edge_y1 >= py1 - 1e-7)
                & (edge_y1 <= py2 + 1e-7)
                & (edge_x1 <= px1 + 1e-7)
                & (edge_x2 >= px1 - 1e-7)
            )

        dx = np.maximum.reduce([px1 - edge_x2, edge_x1 - px2, np.zeros_like(edge_x1)])
        dy = np.maximum.reduce([py1 - edge_y2, edge_y1 - py2, np.zeros_like(edge_y1)])
        clearance_mask = (np.hypot(dx, dy) < required) & ~overlap_mask & ~cross_mask

        for edge_index in np.flatnonzero(overlap_mask):
            u, v, _, _ = edge_list[int(edge_index)]
            blocked_edges.add(normalized_edge(u, v))
        for edge_index in np.flatnonzero(cross_mask):
            u, v, _, _ = edge_list[int(edge_index)]
            edge = normalized_edge(u, v)
            crossing_counts[edge] = crossing_counts.get(edge, 0) + 1
        for edge_index in np.flatnonzero(clearance_mask):
            u, v, _, _ = edge_list[int(edge_index)]
            edge = normalized_edge(u, v)
            clearance_counts[edge] = clearance_counts.get(edge, 0) + 1

    for edge in blocked_edges:
        accumulated_weights[edge] = block_weight

    for edge in set(crossing_counts) | set(clearance_counts):
        if accumulated_weights.get(edge, 0.0) >= block_weight:
            continue
        u, v = edge
        base_dist = float(np.hypot(nodes[v][0] - nodes[u][0], nodes[v][1] - nodes[u][1]))
        base_cost = accumulated_weights.get(edge, base_dist)
        accumulated_weights[edge] = (
            base_cost
            + float(crossing_penalty) * crossing_counts.get(edge, 0)
            + float(clearance_penalty) * clearance_counts.get(edge, 0)
        )
