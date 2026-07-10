from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import combinations, tee
from typing import Callable

import numpy as np

SCORE_PENALIZATION = 1000
TURN_BIAS = 1


@dataclass
class CoreSteinerResult:
    segments: list[tuple[tuple[float, float], tuple[float, float]]]
    tree_nodes: set[int]
    tree_edges: list[tuple[int, int]]
    directed_edges: list[tuple[int, int]]
    directed_segments: list[tuple[tuple[float, float], tuple[float, float]]]
    score: dict[str, float]


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def _heuristic_factory(sx: float, sy: float) -> Callable[[int, int], float]:
    def heuristic(a: int, b: int) -> float:
        ax, ay = _NODE_POSITIONS[a]
        bx, by = _NODE_POSITIONS[b]
        return sx * abs(ax - bx) + sy * abs(ay - by)

    return heuristic


_NODE_POSITIONS: np.ndarray


def _add_edge(graph: dict[int, dict[int, float]], u: int, v: int, weight: float) -> None:
    if u == v:
        return
    graph.setdefault(u, {})
    graph.setdefault(v, {})
    old = graph[u].get(v)
    if old is None or weight < old:
        graph[u][v] = float(weight)
        graph[v][u] = float(weight)


def _edge_items(graph: dict[int, dict[int, float]]) -> list[tuple[int, int, float]]:
    edges = []
    for u, neighbors in graph.items():
        for v, weight in neighbors.items():
            if int(u) < int(v):
                edges.append((int(u), int(v), float(weight)))
    return edges


def _astar_path(
    graph: dict[int, dict[int, float]],
    start: int,
    end: int,
    heuristic: Callable[[int, int], float],
) -> tuple[float, list[int]] | None:
    if start == end:
        return 0.0, [int(start)]
    heap = [(float(heuristic(start, end)), 0.0, int(start), None)]
    best = {int(start): 0.0}
    parent: dict[int, int | None] = {}
    settled: set[int] = set()
    while heap:
        _priority, distance, node, prev = heapq.heappop(heap)
        if node in settled:
            continue
        settled.add(node)
        parent[node] = prev
        if node == end:
            path = []
            curr: int | None = node
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            return float(distance), path
        for nxt, weight in graph.get(node, {}).items():
            if nxt in settled:
                continue
            next_distance = distance + float(weight)
            if next_distance < best.get(nxt, float("inf")):
                best[nxt] = next_distance
                priority = next_distance + float(heuristic(nxt, end))
                heapq.heappush(heap, (priority, next_distance, int(nxt), node))
    return None


class _DisjointSet:
    def __init__(self, nodes):
        self.parent = {int(node): int(node) for node in nodes}
        self.rank = {int(node): 0 for node in nodes}

    def find(self, node: int) -> int:
        node = int(node)
        parent = self.parent.setdefault(node, node)
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _minimum_spanning_tree(graph: dict[int, dict[int, float]]) -> dict[int, dict[int, float]]:
    result = {int(node): {} for node in graph}
    dsu = _DisjointSet(graph.keys())
    for u, v, weight in sorted(_edge_items(graph), key=lambda item: item[2]):
        if dsu.union(u, v):
            _add_edge(result, u, v, weight)
    return result


def _get_complete_distance_graph(
    graph: dict[int, dict[int, float]],
    terminal_nodes: list[int],
    heuristic: Callable[[int, int], float],
) -> dict[int, dict[int, float]] | None:
    complete = {int(node): {} for node in terminal_nodes}
    for n1, n2 in combinations(terminal_nodes, 2):
        solved = _astar_path(graph, int(n1), int(n2), heuristic)
        if solved is None:
            return None
        distance, _path = solved
        _add_edge(complete, int(n1), int(n2), distance)
    return complete


def _remove_redundant_nodes(graph: dict[int, dict[int, float]], terminal_nodes: set[int]) -> dict[int, dict[int, float]]:
    result = {int(node): dict(neighbors) for node, neighbors in graph.items()}
    leaf_non_terminals = [
        node for node in list(result) if len(result.get(node, {})) == 1 and node not in terminal_nodes
    ]
    while leaf_non_terminals:
        node = leaf_non_terminals.pop()
        if node not in result or len(result[node]) != 1:
            continue
        neighbor = next(iter(result[node]))
        result[neighbor].pop(node, None)
        result.pop(node, None)
        if neighbor in result and len(result[neighbor]) == 1 and neighbor not in terminal_nodes:
            leaf_non_terminals.append(neighbor)
    return result


def _same_axis(node_a: int, node_b: int, eps: float = 1e-6) -> bool:
    pa = _NODE_POSITIONS[int(node_a)]
    pb = _NODE_POSITIONS[int(node_b)]
    return abs(float(pa[0] - pb[0])) <= eps or abs(float(pa[1] - pb[1])) <= eps


def _simplify_graph(graph: dict[int, dict[int, float]], terminal_nodes: set[int]) -> dict[int, dict[int, float]]:
    result = {int(node): dict(neighbors) for node, neighbors in graph.items()}
    changed = True
    while changed:
        changed = False
        for node in list(result):
            if node in terminal_nodes or len(result[node]) != 2:
                continue
            neighbors = list(result[node])
            if len(neighbors) != 2:
                continue
            n1, n2 = neighbors
            if not _same_axis(n1, n2):
                continue
            weight = float(result[node][n1]) + float(result[node][n2])
            result[n1].pop(node, None)
            result[n2].pop(node, None)
            result.pop(node, None)
            _add_edge(result, n1, n2, weight)
            changed = True
            break
    return result


def _kou_et_al(
    graph: dict[int, dict[int, float]],
    terminal_nodes: list[int],
    heuristic: Callable[[int, int], float],
) -> dict[int, dict[int, float]] | None:
    if len(graph) <= 1:
        return {int(node): dict(neighbors) for node, neighbors in graph.items()}

    complete = _get_complete_distance_graph(graph, terminal_nodes, heuristic)
    if complete is None or len(_edge_items(_minimum_spanning_tree(complete))) < len(terminal_nodes) - 1:
        return None

    complete_mst = _minimum_spanning_tree(complete)
    expanded: dict[int, dict[int, float]] = {}
    for start, end, _weight in _edge_items(complete_mst):
        solved = _astar_path(graph, start, end, heuristic)
        if solved is None:
            return None
        _distance, path = solved
        for u, v in pairwise(path):
            _add_edge(expanded, int(u), int(v), float(graph[int(u)][int(v)]))

    expanded_mst = _minimum_spanning_tree(expanded)
    return _remove_redundant_nodes(expanded_mst, set(terminal_nodes))


def _valid_edge_sizes(tree: dict[int, dict[int, float]], min_value: float) -> bool:
    if min_value <= 0.0:
        return True
    return all(weight >= min_value for _, _, weight in _edge_items(tree))


def _steiner_min(
    graph: dict[int, dict[int, float]],
    terminal_nodes: list[int],
    min_value: float = 0.0,
) -> tuple[dict[int, dict[int, float]], dict[str, float]] | None:
    directions = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    best_total = float("inf")
    best_tree = None
    best_score = {
        "initial_edges_count": best_total,
        "simplified_edges_count": best_total,
        "turn_bias": float(TURN_BIAS),
        "error_penalization": float(SCORE_PENALIZATION),
        "total": best_total,
    }
    for sx, sy in directions:
        tree = _kou_et_al(graph, terminal_nodes, _heuristic_factory(float(sx), float(sy)))
        if tree is None:
            continue
        initial_edges_count = len(_edge_items(tree))
        simplified = _simplify_graph(tree, set(terminal_nodes))
        simplified_edges_count = len(_edge_items(simplified))
        if simplified_edges_count > 1 and not _valid_edge_sizes(simplified, min_value):
            continue
        total = initial_edges_count + simplified_edges_count * TURN_BIAS
        if total < best_total:
            best_total = float(total)
            best_tree = {int(node): dict(neighbors) for node, neighbors in simplified.items()}
            best_score = {
                "initial_edges_count": float(initial_edges_count),
                "simplified_edges_count": float(simplified_edges_count),
                "turn_bias": float(TURN_BIAS),
                "error_penalization": float(SCORE_PENALIZATION),
                "total": float(total),
            }
    if best_tree is None:
        return None
    return best_tree, best_score


def _build_graph(nodes: np.ndarray, adj: dict[int, list[tuple[int, float, str]]]) -> dict[int, dict[int, float]]:
    graph = {}
    for node_idx in range(len(nodes)):
        graph[int(node_idx)] = {}
    for u, edges in adj.items():
        for v, weight, _direction in edges:
            u_i = int(u)
            v_i = int(v)
            _add_edge(graph, u_i, v_i, float(weight))
    return graph


def _directed_dfs_edges(tree: dict[int, dict[int, float]], root: int) -> list[tuple[int, int]]:
    root = int(root)
    if root not in tree:
        return []
    directed = []
    visited = {root}
    stack = [(root, iter(sorted(int(n) for n in tree[root])))]
    while stack:
        parent, children = stack[-1]
        try:
            child = int(next(children))
        except StopIteration:
            stack.pop()
            continue
        if child in visited:
            continue
        visited.add(child)
        directed.append((parent, child))
        stack.append((child, iter(sorted(int(n) for n in tree.get(child, {})))))
    return directed


def _segments_from_edges(
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segments = []
    for u, v in edges:
        p1 = nodes[int(u)]
        p2 = nodes[int(v)]
        segments.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))
    return segments


def solve_core_steiner_port(
    nodes: np.ndarray,
    adj: dict[int, list[tuple[int, float, str]]],
    terminal_node_indices: list[int],
    min_value: float = 0.0,
) -> CoreSteinerResult | None:
    global _NODE_POSITIONS
    unique_terminals = list(dict.fromkeys(int(node) for node in terminal_node_indices))
    if len(unique_terminals) < 2:
        return None
    _NODE_POSITIONS = np.asarray(nodes, dtype=np.float64)
    graph = _build_graph(_NODE_POSITIONS, adj)
    solved = _steiner_min(graph, unique_terminals, min_value=min_value)
    if solved is None:
        return None
    tree, score = solved
    tree_edges = [(int(u), int(v)) for u, v, _weight in _edge_items(tree)]
    directed_edges = _directed_dfs_edges(tree, unique_terminals[0])
    return CoreSteinerResult(
        segments=_segments_from_edges(_NODE_POSITIONS, tree_edges),
        tree_nodes={int(node) for node in tree},
        tree_edges=tree_edges,
        directed_edges=directed_edges,
        directed_segments=_segments_from_edges(_NODE_POSITIONS, directed_edges),
        score=score,
    )
