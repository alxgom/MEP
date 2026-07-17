"""Fast graph occupancy filters for interactive installation context."""

from __future__ import annotations

import numpy as np


REVERSE_DIRECTION = {"E": "W", "N": "S", "W": "E", "S": "N"}


def adjacency_without_obstacle_bounds(runtime, obstacle_bounds, protected_nodes=()):
    """Eliminate vertices/edges covered by obstacle AABBs without geometry calls."""
    bounds = np.asarray(tuple(obstacle_bounds), dtype=np.float32).reshape((-1, 4))
    if not len(bounds):
        return runtime.adjacency, 0, 0
    protected = np.asarray(tuple(protected_nodes), dtype=np.int64)
    node_x, node_y = runtime.nodes[:, 0], runtime.nodes[:, 1]
    blocked_node_mask = np.zeros(len(runtime.nodes), dtype=bool)
    blocked_edge_mask = np.zeros(len(runtime.edge_list), dtype=bool)
    edge_coords = runtime.edge_coords
    edge_min_x = np.minimum(edge_coords[:, 0], edge_coords[:, 2])
    edge_max_x = np.maximum(edge_coords[:, 0], edge_coords[:, 2])
    edge_min_y = np.minimum(edge_coords[:, 1], edge_coords[:, 3])
    edge_max_y = np.maximum(edge_coords[:, 1], edge_coords[:, 3])
    for min_x, min_y, max_x, max_y in bounds:
        blocked_node_mask |= (
            (node_x >= min_x) & (node_x <= max_x)
            & (node_y >= min_y) & (node_y <= max_y)
        )
        blocked_edge_mask |= (
            (edge_max_x >= min_x) & (edge_min_x <= max_x)
            & (edge_max_y >= min_y) & (edge_min_y <= max_y)
        )
    if len(protected):
        blocked_node_mask[protected] = False
    adjacency = {index: [] for index in range(len(runtime.nodes))}
    for edge_index, (u, v, length, direction) in enumerate(runtime.edge_list):
        if blocked_edge_mask[edge_index] or blocked_node_mask[u] or blocked_node_mask[v]:
            continue
        adjacency[u].append((v, length, direction))
        adjacency[v].append((u, length, REVERSE_DIRECTION[direction]))
    return adjacency, int(blocked_node_mask.sum()), int(blocked_edge_mask.sum())
