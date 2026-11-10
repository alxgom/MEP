"""Experimental rooted, turn-aware supply routing to selectable wall grilles."""

from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math

import numpy as np
from mep_routing.geometry.directions import travel_direction
from mep_routing.routing import SteinerScore, SteinerTreeResult


@dataclass(frozen=True)
class DirectedWallResult:
    nodes: np.ndarray
    adjacency: dict
    tree: SteinerTreeResult
    selected_connections: tuple


@dataclass(frozen=True)
class _PreparedWallGraph:
    nodes: np.ndarray
    adjacency: dict
    candidates: dict
    wall_points: dict


def _direction(a, b):
    return str(travel_direction(a, b))


def _prepare_wall_graph(nodes, adjacency, pairs_by_supply, connections):
    positions = np.asarray(nodes)
    node_at = {tuple(point[:2]): index for index, point in enumerate(positions)}
    candidates, wall_points = {}, {}
    for item in connections:
        name = item.route_name
        choices = {}
        for supply, _returned in pairs_by_supply[name]:
            node = node_at.get(supply.routing_point)
            if node is not None and adjacency.get(node):
                choices.setdefault(node, supply.grille_point)
        if not choices:
            return None
        candidates[name] = tuple(choices)
        wall_points.update(((name, node), point) for node, point in choices.items())
    return _PreparedWallGraph(positions, adjacency, candidates, wall_points)


def _selected_connections(pairs_by_supply, positions, wall_points, selected_nodes):
    selected = []
    for name, node in selected_nodes.items():
        point = wall_points[(name, node)]
        anchor = tuple(positions[node, :2])
        supply = next(
            supply for supply, _returned in pairs_by_supply[name]
            if supply.grille_point == point and supply.routing_point == anchor
        )
        selected.append(replace(supply, placement_type="wall_candidate"))
    return tuple(selected)


def _terminal_cost(position, wall, incoming, bend_penalty):
    length = math.dist(position[:2], wall)
    turns = length > 1e-7 and incoming is not None and incoming != _direction(position, wall)
    return length + (float(bend_penalty) if turns else 0.0)


def solve_directed_wall_tree(nodes, adjacency, root, pairs_by_supply, connections, bend_penalty):
    """Baseline: unite independent turn-aware root-to-terminal paths."""
    prepared = _prepare_wall_graph(nodes, adjacency, pairs_by_supply, connections)
    if prepared is None:
        return None
    positions, result_adj = prepared.nodes, prepared.adjacency
    candidates, wall_points = prepared.candidates, prepared.wall_points
    queue = [(0.0, int(root), "", (int(root), None))]
    best, parent = {(int(root), None): 0.0}, {}
    while queue:
        cost, node, _tie, state = heapq.heappop(queue)
        if cost != best.get(state):
            continue
        incoming = state[1]
        for edge in result_adj.get(node, ()):
            nxt, weight = int(edge[0]), float(edge[1])
            direction = str(edge[2])
            value = cost + weight + (float(bend_penalty) if incoming and incoming != direction else 0.0)
            following = (nxt, direction)
            if value < best.get(following, math.inf):
                best[following] = value
                parent[following] = state
                heapq.heappush(queue, (value, nxt, direction, following))

    union, selected_nodes, selected_costs = set(), {}, []
    for name, choices in candidates.items():
        states = [state for state in best if state[0] in choices]
        if not states:
            return None
        def total(state):
            node, incoming = state
            return best[state] + _terminal_cost(positions[node], wall_points[(name, node)], incoming, bend_penalty)

        state = min(states, key=total)
        selected_costs.append(total(state))
        selected_nodes[name] = state[0]
        while state in parent:
            previous = parent[state]
            union.add(tuple(sorted((state[0], previous[0]))))
            state = previous
    graph = {}
    for u, v in union:
        graph.setdefault(u, set()).add(v)
        graph.setdefault(v, set()).add(u)
    directed, seen, stack = [], {int(root)}, [int(root)]
    while stack:
        u = stack.pop()
        for v in sorted(graph.get(u, ())):
            if v not in seen:
                seen.add(v); directed.append((u, v)); stack.append(v)
    tree = SteinerTreeResult(
        frozenset(seen), tuple(directed), tuple(directed),
        SteinerScore(len(union), len(directed), 1.0, float(sum(selected_costs))),
    )
    selected = _selected_connections(
        pairs_by_supply, positions, wall_points, selected_nodes,
    )
    return DirectedWallResult(positions, result_adj, tree, selected)


def solve_directed_wall_steiner(nodes, adjacency, root, pairs_by_supply, connections, bend_penalty,
                               *, length_policy=None, root_prefix=(), stats=None, fitting_weight=0.0):
    """Exact directed-Steiner subset DP on the turn-aware line graph."""
    if fitting_weight and length_policy is None:
        raise ValueError("fitting perturbation requires the hard-length solver")
    prepared = _prepare_wall_graph(nodes, adjacency, pairs_by_supply, connections)
    if prepared is None:
        return None
    if length_policy is not None:
        from .length_steiner import solve_length_steiner

        solved = solve_length_steiner(
            prepared, root, tuple(item.route_name for item in connections), bend_penalty,
            length_policy, root_prefix=root_prefix, stats=stats, fitting_weight=fitting_weight,
        )
        if solved is None:
            return None
        tree, selected_nodes = solved
        selected = _selected_connections(
            pairs_by_supply, prepared.nodes, prepared.wall_points, selected_nodes,
        )
        return DirectedWallResult(prepared.nodes, prepared.adjacency, tree, selected)
    positions, graph = prepared.nodes, prepared.adjacency
    names = tuple(item.route_name for item in connections)

    states = {(int(root), None)}
    for u, edges in graph.items():
        for edge in edges:
            states.add((int(edge[0]), str(edge[2])))
    states = tuple(sorted(states, key=lambda value: (value[0], value[1] or "")))
    state_index = {state: index for index, state in enumerate(states)}
    states_at_node = {}
    for index, state in enumerate(states):
        states_at_node.setdefault(state[0], []).append(index)

    reverse = [[] for _ in states]
    for index, (node, incoming) in enumerate(states):
        for edge in graph.get(node, ()):
            nxt, direction = int(edge[0]), str(edge[2])
            following = state_index.get((nxt, direction))
            if following is None:
                continue
            cost = float(edge[1]) + (
                float(bend_penalty) if incoming is not None and incoming != direction else 0.0
            )
            reverse[following].append((index, cost, node, nxt))

    count, full = len(names), (1 << len(names)) - 1
    costs = [dict() for _ in range(full + 1)]
    choices = [dict() for _ in range(full + 1)]
    for bit, name in enumerate(names):
        mask = 1 << bit
        for node in prepared.candidates[name]:
            wall = prepared.wall_points[(name, node)]
            for state in states_at_node.get(node, ()):
                value = _terminal_cost(positions[node], wall, states[state][1], bend_penalty)
                if value < costs[mask].get(state, math.inf):
                    costs[mask][state] = value
                    choices[mask][state] = ("terminal", name, node)

    for mask in range(1, full + 1):
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if other and sub < other:
                left, right = costs[sub], costs[other]
                for state in left.keys() & right.keys():
                    value = left[state] + right[state]
                    if value < costs[mask].get(state, math.inf):
                        costs[mask][state] = value
                        choices[mask][state] = ("merge", sub, other)
            sub = (sub - 1) & mask

        queue = [(value, state) for state, value in costs[mask].items()]
        heapq.heapify(queue)
        while queue:
            value, state = heapq.heappop(queue)
            if value != costs[mask].get(state):
                continue
            for previous, arc_cost, u, v in reverse[state]:
                candidate = value + arc_cost
                if candidate < costs[mask].get(previous, math.inf):
                    costs[mask][previous] = candidate
                    choices[mask][previous] = ("arc", state, u, v)
                    heapq.heappush(queue, (candidate, previous))

    root_state = state_index[(int(root), None)]
    if root_state not in costs[full]:
        return None
    arcs, selected_nodes = set(), {}

    def reconstruct(mask, state):
        choice = choices[mask][state]
        if choice[0] == "terminal":
            selected_nodes[choice[1]] = choice[2]
        elif choice[0] == "merge":
            reconstruct(choice[1], state)
            reconstruct(choice[2], state)
        else:
            arcs.add((choice[2], choice[3]))
            reconstruct(mask, choice[1])

    reconstruct(full, root_state)
    indegree = {}
    for u, v in arcs:
        indegree[v] = indegree.get(v, 0) + 1
    if any(value > 1 for value in indegree.values()):
        return None
    directed = tuple(sorted(arcs))
    tree = SteinerTreeResult(
        frozenset({int(root)} | {node for edge in directed for node in edge}),
        directed, directed,
        SteinerScore(len(directed), len(directed), 1.0, float(costs[full][root_state])),
    )
    selected = _selected_connections(
        pairs_by_supply, positions, prepared.wall_points, selected_nodes,
    )
    return DirectedWallResult(positions, graph, tree, selected)
