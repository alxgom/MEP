from __future__ import annotations

import math

from mep_routing.geometry import normalize_axis_segment


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
