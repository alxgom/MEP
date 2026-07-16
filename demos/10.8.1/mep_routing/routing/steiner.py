"""Installation-neutral approximate Steiner-tree routing on an indexed graph."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import combinations, pairwise
import math
from typing import Iterable, Sequence

import numpy as np


DEFAULT_HEURISTIC_DIRECTIONS = (
    (0.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (1.0, 1.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (-1.0, -1.0),
)


@dataclass(frozen=True)
class SteinerScore:
    initial_edge_count: int
    simplified_edge_count: int
    turn_bias: float
    total: float


@dataclass(frozen=True)
class SteinerTreeResult:
    tree_nodes: frozenset[int]
    tree_edges: tuple[tuple[int, int], ...]
    directed_edges: tuple[tuple[int, int], ...]
    score: SteinerScore


Graph = dict[int, dict[int, float]]


def segments_from_edges(nodes, edges: Iterable[tuple[int, int]]):
    """Convert indexed edges to immutable world-coordinate segments."""
    positions = np.asarray(nodes)
    return tuple(
        (
            (float(positions[u][0]), float(positions[u][1])),
            (float(positions[v][0]), float(positions[v][1])),
        )
        for u, v in edges
    )


def _add_edge(graph: Graph, u: int, v: int, weight: float) -> None:
    if u == v:
        return
    old = graph[u].get(v)
    if old is None or weight < old:
        graph[u][v] = weight
        graph[v][u] = weight


def _edges(graph: Graph):
    return tuple(
        sorted(
            (u, v, weight)
            for u, neighbours in graph.items()
            for v, weight in neighbours.items()
            if u < v
        )
    )


def _normalized_graph(nodes: np.ndarray, adjacency) -> Graph:
    graph = {index: {} for index in range(len(nodes))}
    for raw_u, neighbours in adjacency.items():
        u = int(raw_u)
        if u not in graph:
            raise ValueError(f"adjacency node index out of range: {u}")
        for edge in neighbours:
            if len(edge) < 2:
                raise ValueError("adjacency edges require a node index and weight")
            v, weight = int(edge[0]), float(edge[1])
            if v not in graph:
                raise ValueError(f"adjacency node index out of range: {v}")
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError("edge weights must be finite and non-negative")
            _add_edge(graph, u, v, weight)
    return graph


def _validate_nodes(nodes) -> np.ndarray:
    positions = np.asarray(nodes, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("nodes must be a two-dimensional array with x/y coordinates")
    if not np.isfinite(positions[:, :2]).all():
        raise ValueError("node coordinates must be finite")
    return positions


def _shortest_path(graph: Graph, positions: np.ndarray, start: int, end: int, scale):
    if start == end:
        return 0.0, (start,)

    sx, sy = scale

    def heuristic(node):
        return sx * abs(positions[node][0] - positions[end][0]) + sy * abs(positions[node][1] - positions[end][1])

    queue = [(float(heuristic(start)), 0.0, start)]
    distances = {start: 0.0}
    parents = {}
    settled = set()
    while queue:
        _priority, distance, node = heapq.heappop(queue)
        if node in settled:
            continue
        settled.add(node)
        if node == end:
            path = [node]
            while node != start:
                node = parents[node]
                path.append(node)
            return distance, tuple(reversed(path))
        for neighbour, weight in sorted(graph[node].items()):
            candidate = distance + weight
            if neighbour in settled or candidate >= distances.get(neighbour, math.inf):
                continue
            distances[neighbour] = candidate
            parents[neighbour] = node
            heapq.heappush(queue, (candidate + float(heuristic(neighbour)), candidate, neighbour))
    return None


class _DisjointSet:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}

    def find(self, node):
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        return True


def _minimum_spanning_tree(graph: Graph) -> Graph:
    tree = {node: {} for node in graph}
    sets = _DisjointSet(graph)
    for u, v, weight in sorted(_edges(graph), key=lambda edge: (edge[2], edge[0], edge[1])):
        if sets.union(u, v):
            _add_edge(tree, u, v, weight)
    return tree


def _prune_leaves(graph: Graph, terminals: frozenset[int]) -> Graph:
    tree = {node: dict(neighbours) for node, neighbours in graph.items()}
    pending = sorted(node for node in tree if len(tree[node]) == 1 and node not in terminals)
    while pending:
        node = pending.pop()
        if node not in tree or len(tree[node]) != 1:
            continue
        neighbour = next(iter(tree[node]))
        tree[neighbour].pop(node)
        tree.pop(node)
        if len(tree[neighbour]) == 1 and neighbour not in terminals:
            pending.append(neighbour)
    return tree


def _simplify(tree: Graph, terminals: frozenset[int], positions: np.ndarray) -> Graph:
    result = {node: dict(neighbours) for node, neighbours in tree.items()}
    changed = True
    while changed:
        changed = False
        for node in sorted(result):
            if node in terminals or len(result[node]) != 2:
                continue
            left, right = sorted(result[node])
            p_left, p_right = positions[left], positions[right]
            same_axis = abs(p_left[0] - p_right[0]) <= 1e-6 or abs(p_left[1] - p_right[1]) <= 1e-6
            if not same_axis:
                continue
            weight = result[node][left] + result[node][right]
            result[left].pop(node)
            result[right].pop(node)
            result.pop(node)
            _add_edge(result, left, right, weight)
            changed = True
            break
    return result


def _kou_tree(graph: Graph, terminals: tuple[int, ...], positions: np.ndarray, scale):
    complete = {terminal: {} for terminal in terminals}
    paths = {}
    for left, right in combinations(terminals, 2):
        solved = _shortest_path(graph, positions, left, right, scale)
        if solved is None:
            return None
        distance, path = solved
        _add_edge(complete, left, right, distance)
        paths[left, right] = path

    closure_tree = _minimum_spanning_tree(complete)
    if len(_edges(closure_tree)) != len(terminals) - 1:
        return None
    expanded: Graph = {node: {} for node in graph}
    for left, right, _weight in _edges(closure_tree):
        path = paths.get((left, right))
        if path is None:
            solved = _shortest_path(graph, positions, left, right, scale)
            if solved is None:
                return None
            _distance, path = solved
        for u, v in pairwise(path):
            _add_edge(expanded, u, v, graph[u][v])
    return _prune_leaves(_minimum_spanning_tree(expanded), frozenset(terminals))


def _directed_edges(tree: Graph, root: int):
    directed = []
    visited = {root}
    stack = [(root, iter(sorted(tree[root])))]
    while stack:
        parent, children = stack[-1]
        try:
            child = next(children)
        except StopIteration:
            stack.pop()
            continue
        if child in visited:
            continue
        visited.add(child)
        directed.append((parent, child))
        stack.append((child, iter(sorted(tree[child]))))
    return tuple(directed)


def solve_steiner_tree(
    nodes,
    adjacency,
    terminal_node_indices: Sequence[int],
    *,
    min_edge_size: float = 0.0,
    heuristic_directions: Sequence[tuple[float, float]] = DEFAULT_HEURISTIC_DIRECTIONS,
) -> SteinerTreeResult | None:
    """Approximate a minimum rectilinear tree connecting all terminal nodes.

    The first unique terminal is the root used to orient ``directed_edges``.
    ``None`` means fewer than two unique terminals or no feasible tree.
    """
    positions = _validate_nodes(nodes)
    if not math.isfinite(min_edge_size) or min_edge_size < 0.0:
        raise ValueError("min_edge_size must be finite and non-negative")
    terminals = tuple(dict.fromkeys(int(node) for node in terminal_node_indices))
    if any(node < 0 or node >= len(positions) for node in terminals):
        raise ValueError("terminal node index out of range")
    if len(terminals) < 2:
        return None
    directions = tuple((float(x), float(y)) for x, y in heuristic_directions)
    if not directions or any(not math.isfinite(value) for direction in directions for value in direction):
        raise ValueError("heuristic directions must contain finite scale pairs")

    graph = _normalized_graph(positions, adjacency)
    best = None
    best_key = None
    terminals_set = frozenset(terminals)
    for order, scale in enumerate(directions):
        initial = _kou_tree(graph, terminals, positions, scale)
        if initial is None:
            continue
        initial_count = len(_edges(initial))
        simplified = _simplify(initial, terminals_set, positions)
        edges = _edges(simplified)
        if any(weight < min_edge_size for _u, _v, weight in edges):
            continue
        tree_edges = tuple((u, v) for u, v, _weight in edges)
        total = float(initial_count + len(tree_edges))
        key = (total, tree_edges, order)
        if best_key is None or key < best_key:
            best_key = key
            best = SteinerTreeResult(
                tree_nodes=frozenset(node for edge in tree_edges for node in edge),
                tree_edges=tree_edges,
                directed_edges=_directed_edges(simplified, terminals[0]),
                score=SteinerScore(initial_count, len(tree_edges), 1.0, total),
            )
    return best
