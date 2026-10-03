"""Experimental rooted, turn-aware supply routing to selectable wall grilles."""

from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math

import numpy as np
from shapely.geometry import LineString, Point

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
    dx, dy = b[0] - a[0], b[1] - a[1]
    return ("E" if dx >= 0 else "W") if abs(dx) >= abs(dy) else ("S" if dy >= 0 else "N")


def _corner_clear(boundary, point, clearance):
    distance = boundary.project(point)
    coordinates = list(boundary.coords)
    walked = 0.0
    for left, right in zip(coordinates, coordinates[1:]):
        if abs(distance - walked) < clearance:
            return False
        walked += math.dist(left[:2], right[:2])
    return abs(distance - walked) >= clearance


def _inward_normal(room, point):
    coordinates = list(room.polygon.exterior.coords)
    nearest = min(
        zip(coordinates, coordinates[1:]),
        key=lambda pair: LineString(pair).distance(Point(point)),
    )
    dx, dy = nearest[1][0] - nearest[0][0], nearest[1][1] - nearest[0][1]
    length = max(math.hypot(dx, dy), 1e-9)
    normals = ((-dy / length, dx / length), (dy / length, -dx / length))
    return next(
        (normal for normal in normals if room.polygon.contains(Point(
            point[0] + normal[0] * 20.0, point[1] + normal[1] * 20.0,
        ))),
        normals[0],
    )


def _candidate_specs(
    nodes, adjacency, rooms_by_name, connections, margin_mm=50.0,
    maximum_projection_mm=600.0,
):
    specs = {item.route_name: [] for item in connections}
    for u, neighbors in adjacency.items():
        for raw in neighbors:
            v = int(raw[0])
            if int(u) >= v:
                continue
            edge = LineString((nodes[int(u), :2], nodes[v, :2]))
            for item in connections:
                room = rooms_by_name.get(item.room_name)
                if room is None:
                    continue
                boundary = room.polygon.exterior
                hit = edge.intersection(boundary)
                # Grid edges commonly terminate exactly on a generated wall line;
                # that is a valid crossing candidate even though GEOS classifies
                # it as a touch rather than a strict ``crosses`` relation.
                if not isinstance(hit, Point):
                    continue
                if not _corner_clear(boundary, hit, item.width_mm / 2.0 + margin_mm):
                    continue
                specs[item.route_name].append((int(u), v, (round(hit.x), round(hit.y))))
    # Most clipped routing graphs stop short of the wall.  Treat a nearby node
    # as a crossing of its incident graph axis when that axis is normal to the
    # wall projection.  The projected point is the physical grille and the
    # existing node remains the routing endpoint.
    for item in connections:
        if specs[item.route_name]:
            continue
        room = rooms_by_name.get(item.room_name)
        if room is None:
            continue
        boundary = room.polygon.exterior
        for node_index, raw_point in enumerate(nodes[:, :2]):
            point = Point(float(raw_point[0]), float(raw_point[1]))
            projected = boundary.interpolate(boundary.project(point))
            distance = point.distance(projected)
            if distance <= 1e-7 or distance > maximum_projection_mm:
                continue
            wall_point = (round(projected.x), round(projected.y))
            if not _corner_clear(boundary, projected, item.width_mm / 2.0 + margin_mm):
                continue
            vx, vy = projected.x - point.x, projected.y - point.y
            vlength = max(math.hypot(vx, vy), 1e-9)
            aligned = False
            for edge in adjacency.get(node_index, ()):
                neighbor = nodes[int(edge[0]), :2]
                ex, ey = neighbor[0] - raw_point[0], neighbor[1] - raw_point[1]
                elength = max(math.hypot(ex, ey), 1e-9)
                if abs((ex * vx + ey * vy) / (elength * vlength)) >= 0.95:
                    aligned = True
                    break
            if aligned:
                specs[item.route_name].append((node_index, node_index, wall_point))
    return specs


def _prepare_wall_graph(nodes, adjacency, rooms_by_name, connections):
    nodes = np.asarray(nodes)
    specs = _candidate_specs(nodes, adjacency, rooms_by_name, connections)
    if any(not specs[item.route_name] for item in connections):
        return None

    result_nodes = nodes.tolist()
    result_adj = {int(u): list(edges) for u, edges in adjacency.items()}
    candidates, wall_points = {}, {}
    for item in connections:
        candidates[item.route_name] = []
        for u, v, point in specs[item.route_name]:
            if u == v:
                candidates[item.route_name].append(u)
                wall_points[(item.route_name, u)] = point
                continue
            index = len(result_nodes)
            result_nodes.append(point)
            candidates[item.route_name].append(index)
            wall_points[(item.route_name, index)] = point
            original = next((edge for edge in result_adj[u] if int(edge[0]) == v), None)
            if original is None:
                continue
            reverse = next(edge for edge in result_adj[v] if int(edge[0]) == u)
            result_adj[u] = [edge for edge in result_adj[u] if int(edge[0]) != v]
            result_adj[v] = [edge for edge in result_adj[v] if int(edge[0]) != u]
            total = max(math.dist(nodes[u, :2], nodes[v, :2]), 1e-9)
            left = float(original[1]) * math.dist(nodes[u, :2], point) / total
            right = float(original[1]) * math.dist(point, nodes[v, :2]) / total
            result_adj[index] = [(u, left, _direction(point, nodes[u])), (v, right, _direction(point, nodes[v]))]
            result_adj[u].append((index, left, _direction(nodes[u], point)))
            result_adj[v].append((index, right, _direction(nodes[v], point)))

    return _PreparedWallGraph(
        np.asarray(result_nodes, dtype=np.float64), result_adj,
        candidates, wall_points,
    )


def _selected_connections(connections, rooms_by_name, positions, wall_points, selected_nodes):
    by_name = {item.route_name: item for item in connections}
    selected = []
    for name, node in selected_nodes.items():
        item = by_name[name]
        point = wall_points[(name, node)]
        room = rooms_by_name[item.room_name]
        selected.append(replace(
            item, grille_point=point,
            routing_point=tuple(map(int, positions[node, :2])),
            inward_normal=_inward_normal(room, point), placement_type="wall_candidate",
        ))
    return tuple(selected)


def solve_directed_wall_tree(nodes, adjacency, root, rooms_by_name, connections, bend_penalty):
    """Baseline: unite independent turn-aware root-to-terminal paths."""
    prepared = _prepare_wall_graph(nodes, adjacency, rooms_by_name, connections)
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
        state = min(states, key=lambda value: best[value])
        selected_costs.append(best[state])
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
        connections, rooms_by_name, positions, wall_points, selected_nodes,
    )
    return DirectedWallResult(positions, result_adj, tree, selected)


def solve_directed_wall_steiner(nodes, adjacency, root, rooms_by_name, connections, bend_penalty,
                               *, length_policy=None, root_prefix=(), stats=None):
    """Exact directed-Steiner subset DP on the turn-aware line graph."""
    prepared = _prepare_wall_graph(nodes, adjacency, rooms_by_name, connections)
    if prepared is None:
        return None
    if length_policy is not None:
        from .length_steiner import solve_length_steiner

        solved = solve_length_steiner(
            prepared, root, tuple(item.route_name for item in connections), bend_penalty,
            length_policy, root_prefix=root_prefix, stats=stats,
        )
        if solved is None:
            return None
        tree, selected_nodes = solved
        selected = _selected_connections(
            connections, rooms_by_name, prepared.nodes, prepared.wall_points, selected_nodes,
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
            connector_direction = _direction(positions[node], wall)
            connector_length = math.dist(positions[node, :2], wall)
            for state in states_at_node.get(node, ()):
                incoming = states[state][1]
                value = connector_length + (
                    float(bend_penalty)
                    if connector_length > 1e-7 and incoming is not None
                    and incoming != connector_direction else 0.0
                )
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
        connections, rooms_by_name, positions, prepared.wall_points, selected_nodes,
    )
    return DirectedWallResult(positions, graph, tree, selected)
