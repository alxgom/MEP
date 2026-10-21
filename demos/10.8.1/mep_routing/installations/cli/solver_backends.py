"""Cli-owned multi-terminal solver backend dispatch."""

from __future__ import annotations

import heapq
import math
from itertools import combinations

from shapely.geometry import LineString

from mep_routing.routing import SteinerScore, SteinerTreeResult, solve_steiner_tree
from mep_routing.geometry.directions import same_direction


ALL_HEURISTIC_DIRECTIONS = (
    (0, 0), (0, 1), (0, -1), (1, 0), (-1, 0),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)
NONNEGATIVE_HEURISTIC_DIRECTIONS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _edge_direction(nodes, u, v):
    dx = float(nodes[v][0]) - float(nodes[u][0])
    dy = float(nodes[v][1]) - float(nodes[u][1])
    if abs(dx) >= abs(dy):
        return "E" if dx >= 0 else "W"
    return "S" if dy >= 0 else "N"


def _crossing_edges(nodes, adjacency, prior_segments):
    prior = tuple(LineString(segment) for segment in prior_segments)
    if not prior:
        return frozenset()
    crossed = set()
    for u, neighbors in adjacency.items():
        for edge in neighbors:
            v = int(edge[0])
            if int(u) >= v:
                continue
            line = LineString((nodes[int(u)][:2], nodes[v][:2]))
            if any(line.crosses(other) for other in prior):
                crossed.add((int(u), v))
    return frozenset(crossed)


def _bend_path(nodes, adjacency, start, target, bend_penalty, crossing_penalty, crossed):
    start_state = (int(start), None)
    queue = [(0.0, int(start), "", start_state)]
    best = {start_state: 0.0}
    parent = {}
    final = None
    while queue:
        cost, node, _tie, state = heapq.heappop(queue)
        if cost != best.get(state):
            continue
        if node == int(target):
            final = state
            break
        incoming = state[1]
        for edge in adjacency.get(node, ()):
            nxt, distance = int(edge[0]), float(edge[1])
            direction = edge[2] if len(edge) > 2 else _edge_direction(nodes, node, nxt)
            turn = bend_penalty if incoming is not None and not same_direction(direction, incoming) else 0.0
            key = (min(node, nxt), max(node, nxt))
            candidate = cost + distance + turn + (crossing_penalty if key in crossed else 0.0)
            next_state = (nxt, direction)
            if candidate < best.get(next_state, math.inf):
                best[next_state] = candidate
                parent[next_state] = state
                heapq.heappush(queue, (candidate, nxt, str(direction), next_state))
    if final is None:
        return None
    path = []
    state = final
    while True:
        path.append(state[0])
        if state == start_state:
            break
        state = parent[state]
    return best[final], tuple(reversed(path))


class _DisjointSet:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return False
        self.parent[right] = left
        return True


def solve_line_graph_mst(
    nodes, adjacency, terminal_node_indices, *, bend_penalty=0.0,
    crossing_penalty=0.0, prior_segments=(),
):
    """Connect terminals through a bend-aware metric-closure MST."""

    terminals = tuple(dict.fromkeys(int(node) for node in terminal_node_indices))
    if len(terminals) < 2:
        return None
    crossed = _crossing_edges(nodes, adjacency, prior_segments)
    metric = []
    for left, right in combinations(terminals, 2):
        solved = _bend_path(
            nodes, adjacency, left, right, float(bend_penalty),
            float(crossing_penalty), crossed,
        )
        if solved is not None:
            cost, path = solved
            metric.append((cost, left, right, path))
    dsu = _DisjointSet(terminals)
    selected = []
    for item in sorted(metric, key=lambda value: (value[0], value[1], value[2])):
        if dsu.union(item[1], item[2]):
            selected.append(item)
    if len(selected) != len(terminals) - 1:
        return None
    union_edges = set()
    for _cost, _left, _right, path in selected:
        union_edges.update((min(u, v), max(u, v)) for u, v in zip(path, path[1:]))
    weighted = sorted(
        (math.dist(nodes[u][:2], nodes[v][:2]), u, v) for u, v in union_edges
    )
    tree_dsu = _DisjointSet({node for edge in union_edges for node in edge})
    tree_edges = []
    for _weight, u, v in weighted:
        if tree_dsu.union(u, v):
            tree_edges.append((u, v))
    graph = {}
    for u, v in tree_edges:
        graph.setdefault(u, set()).add(v)
        graph.setdefault(v, set()).add(u)
    directed, visited, stack = [], {terminals[0]}, [terminals[0]]
    while stack:
        parent = stack.pop()
        for child in sorted(graph.get(parent, ()), reverse=True):
            if child in visited:
                continue
            visited.add(child)
            directed.append((parent, child))
            stack.append(child)
    total = sum(item[0] for item in selected)
    return SteinerTreeResult(
        tree_nodes=frozenset(graph), tree_edges=tuple(tree_edges),
        directed_edges=tuple(directed),
        score=SteinerScore(len(union_edges), len(tree_edges), 1.0, float(total)),
    )


def solve_cli_tree(
    backend_index, nodes, adjacency, terminals, *, bend_penalty=0.0,
    crossing_multiplier=0.0, prior_segments=(),
    euclidean=False,
):
    if backend_index == 0:
        return solve_steiner_tree(
            nodes, adjacency, terminals,
            heuristic_directions=ALL_HEURISTIC_DIRECTIONS,
            euclidean=euclidean,
        )
    if backend_index == 1:
        return solve_steiner_tree(
            nodes, adjacency, terminals,
            heuristic_directions=NONNEGATIVE_HEURISTIC_DIRECTIONS,
            euclidean=euclidean,
        )
    if backend_index == 2:
        return solve_line_graph_mst(
            nodes, adjacency, terminals, bend_penalty=bend_penalty,
            crossing_penalty=float(crossing_multiplier) * float(bend_penalty),
            prior_segments=prior_segments,
        )
    raise ValueError(f"unknown Cli solver backend: {backend_index}")
