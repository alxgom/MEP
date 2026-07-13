from __future__ import annotations

import heapq

import numpy as np


def compute_dijkstra_distance_field(start_nodes, env):
    if isinstance(start_nodes, (int, np.integer)):
        start_nodes = [start_nodes]
    distances = {n: 1e9 for n in env.adj}
    pq = []
    for n in start_nodes:
        distances[n] = 0.0
        heapq.heappush(pq, (0.0, n))

    while pq:
        dist, u = heapq.heappop(pq)
        if dist > distances[u]:
            continue
        for v, edge_dist, _direction in env.adj.get(u, []):
            new_dist = dist + edge_dist
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    return distances


def placement_weights(weight_mode_idx):
    if weight_mode_idx == 1:
        return {
            "Shaft": 1.0,
            "Kitchen": 1.0,
            "Bathroom": 1.0,
            "Bathroom 1": 1.0,
            "Bathroom 2": 1.0,
            "Toilet": 1.0,
            "Washroom": 1.0,
        }
    return {
        "Shaft": 2.5,
        "Kitchen": 1.5,
        "Bathroom": 1.0,
        "Bathroom 1": 1.0,
        "Bathroom 2": 1.0,
        "Toilet": 1.0,
        "Washroom": 1.0,
    }


def topological_placement_scores(env, shaft_boundary_nodes, terminal_nodes, weights):
    distance_fields = {
        "Shaft": compute_dijkstra_distance_field(shaft_boundary_nodes, env),
    }
    for name, node_idx in terminal_nodes.items():
        distance_fields[name] = compute_dijkstra_distance_field(node_idx, env)

    node_scores = {}
    for n in range(len(env.nodes)):
        total_score = 0.0
        reachable = True
        for name, field in distance_fields.items():
            dist = field.get(n, 1e9)
            if dist >= 1e8:
                reachable = False
                break
            total_score += weights.get(name, 1.0) * dist

        if reachable:
            node_scores[n] = total_score

    return node_scores, distance_fields
