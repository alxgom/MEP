from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point
from shapely.prepared import prep

from .env import EnvView


def filter_dynamic_machine_obstacle(nodes, edge_list, edge_coords, machine_polygon, clearance_mm, protected_nodes=()):
    """Filter a static graph against the current machine obstacle and return a graph view."""
    blocked_polygon = machine_polygon.buffer(clearance_mm, join_style=2)
    min_x, min_y, max_x, max_y = blocked_polygon.bounds
    protected_nodes = set(protected_nodes)
    prepared_obstacle = prep(blocked_polygon)

    node_x, node_y = nodes[:, 0], nodes[:, 1]
    candidate_nodes = np.where((node_x >= min_x) & (node_x <= max_x) & (node_y >= min_y) & (node_y <= max_y))[0]
    blocked_nodes = {
        int(index)
        for index in candidate_nodes
        if int(index) not in protected_nodes and prepared_obstacle.contains(Point(float(nodes[index, 0]), float(nodes[index, 1])))
    }

    segment_min_x = np.minimum(edge_coords[:, 0], edge_coords[:, 2])
    segment_max_x = np.maximum(edge_coords[:, 0], edge_coords[:, 2])
    segment_min_y = np.minimum(edge_coords[:, 1], edge_coords[:, 3])
    segment_max_y = np.maximum(edge_coords[:, 1], edge_coords[:, 3])
    candidate_edges = np.where((segment_max_x >= min_x) & (segment_min_x <= max_x) & (segment_max_y >= min_y) & (segment_min_y <= max_y))[0]
    blocked_edges = set()
    for edge_index in candidate_edges:
        u, v, _length, _direction = edge_list[edge_index]
        if u in blocked_nodes or v in blocked_nodes:
            blocked_edges.add(int(edge_index))
            continue
        line = LineString([(float(nodes[u, 0]), float(nodes[u, 1])), (float(nodes[v, 0]), float(nodes[v, 1]))])
        intersection = line.intersection(blocked_polygon)
        if not intersection.is_empty and intersection.length > 1.0 and u not in protected_nodes and v not in protected_nodes:
            blocked_edges.add(int(edge_index))

    reverse_direction = {"E": "W", "N": "S", "W": "E", "S": "N"}
    adjacency = {index: [] for index in range(len(nodes))}
    for edge_index, (u, v, length, direction) in enumerate(edge_list):
        if edge_index in blocked_edges or u in blocked_nodes or v in blocked_nodes:
            continue
        adjacency[u].append((v, length, direction))
        adjacency[v].append((u, length, reverse_direction[direction]))
    return EnvView(nodes, adjacency), len(blocked_nodes), len(blocked_edges)
