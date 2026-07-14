from __future__ import annotations

import heapq
from dataclasses import dataclass
from numbers import Integral
from typing import Generic, TypeVar

from .contracts import RoutingProblem, SolvedRoute, SolverFailure, SolverResult


Target = TypeVar("Target")


@dataclass(frozen=True)
class PinFlowRoute(Generic[Target]):
    """A routed graph path and the selected eligible pin target."""

    path: tuple[int, ...]
    target: Target


def add_edge(graph, u, v, cap, cost, meta=None):
    fwd = {"to": v, "rev": len(graph[v]), "cap": cap, "orig_cap": cap, "cost": float(cost), "meta": meta}
    rev = {"to": u, "rev": len(graph[u]), "cap": 0, "orig_cap": 0, "cost": -float(cost), "meta": None}
    graph[u].append(fwd)
    graph[v].append(rev)


def min_cost_flow(graph, source, sink, flow_required):
    flow = 0
    cost = 0.0
    potentials = [0.0] * len(graph)

    while flow < flow_required:
        dist = [float("inf")] * len(graph)
        parent = [None] * len(graph)
        dist[source] = 0.0
        pq = [(0.0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u] + 1e-9:
                continue
            for ei, edge in enumerate(graph[u]):
                if edge["cap"] <= 0:
                    continue
                v = edge["to"]
                nd = d + edge["cost"] + potentials[u] - potentials[v]
                if nd + 1e-9 < dist[v]:
                    dist[v] = nd
                    parent[v] = (u, ei)
                    heapq.heappush(pq, (nd, v))

        if parent[sink] is None:
            break

        for i, d in enumerate(dist):
            if d < float("inf"):
                potentials[i] += d

        add = flow_required - flow
        v = sink
        while v != source:
            u, ei = parent[v]
            add = min(add, graph[u][ei]["cap"])
            v = u

        v = sink
        while v != source:
            u, ei = parent[v]
            edge = graph[u][ei]
            edge["cap"] -= add
            graph[v][edge["rev"]]["cap"] += add
            cost += add * edge["cost"]
            v = u

        flow += add

    return flow, cost


def positive_flow_edges(graph, u):
    return [
        edge
        for edge in graph[u]
        if edge["orig_cap"] > 0 and edge["orig_cap"] - edge["cap"] > 0
    ]


def trace_flow_path(graph, start_node, sink):
    states = []
    target = None
    u = start_node
    seen = set()

    while u != sink:
        if u in seen:
            return None, None
        seen.add(u)

        candidates = positive_flow_edges(graph, u)
        if not candidates:
            return None, None
        edge = candidates[0]
        edge["cap"] += 1

        meta = edge.get("meta")
        if meta:
            if meta[0] == "state":
                states.append((meta[1], meta[2]))
            elif meta[0] == "target":
                target = meta[1]
        u = edge["to"]

    if not states or target is None:
        return None, None
    path = [states[0][0]]
    path.extend(v for _, v in states)
    return path, target


def source_start_nodes(source_spec, kd):
    if isinstance(source_spec, (list, tuple, set)):
        values = list(source_spec)
        if not values:
            return []
        if isinstance(values[0], Integral):
            return [int(v) for v in values]
    _, start_idx = kd.query(source_spec)
    return [int(start_idx)]


def pin_target_specs(route_names, pin_node_map, eligible_pins):
    return {
        route_name: [
            target
            for pin_name in eligible_pins
            for target in pin_node_map.get(pin_name, [])
        ]
        for route_name in route_names
    }


def small_pin_target_specs(room_names, pin_node_map, small_pins=("tl", "tr", "bl", "br")):
    """Compatibility adapter for Sal's current small-port topology."""
    return pin_target_specs(room_names, pin_node_map, small_pins)


def build_pin_min_cost_flow_network(
    route_names,
    target_specs_by_route,
    start_nodes_by_route,
    adjacency,
    edge_cost_fn,
    direction_fn,
    bend_penalty,
    overlap_block_weight,
):
    """Build the shared line-graph residual network used by pin-routing flow."""
    all_targets = [
        target
        for route_name in route_names
        for target in target_specs_by_route.get(route_name, [])
    ]
    if not all_targets:
        return None

    source = 0
    sink = 1
    graph = [[] for _ in range(2)]

    def new_node():
        graph.append([])
        return len(graph) - 1

    route_flow_nodes = {}
    for route_name in route_names:
        route_flow_nodes[route_name] = new_node()
        add_edge(graph, source, route_flow_nodes[route_name], 1, 0.0)

    state_nodes = {}
    for u, edges in adjacency.items():
        for v, _dist, _direction in edges:
            state_nodes[(int(u), int(v))] = (new_node(), new_node())

    extra_state_capacity = max(len(route_names) - 1, 0)
    for (u, v), (state_in, state_out) in state_nodes.items():
        add_edge(graph, state_in, state_out, 1, 0.0, ("state", u, v))
        if extra_state_capacity:
            add_edge(graph, state_in, state_out, extra_state_capacity, overlap_block_weight, ("state", u, v))

    for route_name in route_names:
        for start_idx in start_nodes_by_route.get(route_name, []):
            for v, dist, _direction in adjacency.get(start_idx, []):
                v = int(v)
                if (start_idx, v) not in state_nodes:
                    continue
                edge_cost = edge_cost_fn(start_idx, v, dist)
                state_in, _ = state_nodes[(start_idx, v)]
                add_edge(graph, route_flow_nodes[route_name], state_in, 1, edge_cost)

    for (u, v), (_state_in, state_out) in state_nodes.items():
        current_direction = direction_fn(u, v)
        for w, dist, next_direction in adjacency.get(v, []):
            w = int(w)
            if w == u or (v, w) not in state_nodes:
                continue
            next_in, _ = state_nodes[(v, w)]
            edge_cost = edge_cost_fn(v, w, dist)
            turn_penalty = bend_penalty if current_direction != next_direction else 0.0
            add_edge(graph, state_out, next_in, len(route_names), edge_cost + turn_penalty)

    pin_nodes = {}
    for target in all_targets:
        pin_nodes.setdefault(target["pin"], new_node())
    for pin_node in pin_nodes.values():
        add_edge(graph, pin_node, sink, 1, 0.0)

    for route_name in route_names:
        for target in target_specs_by_route.get(route_name, []):
            target_node = int(target["node_idx"])
            spec_node = new_node()
            add_edge(graph, spec_node, pin_nodes[target["pin"]], 1, 0.0, ("target", target))

            for u in adjacency:
                u = int(u)
                if (u, target_node) not in state_nodes:
                    continue
                _, state_out = state_nodes[(u, target_node)]
                current_direction = direction_fn(u, target_node)
                final_penalty = bend_penalty if current_direction != target["in_dir"] else 0.0
                add_edge(graph, state_out, spec_node, 1, final_penalty)

    return graph, source, sink, route_flow_nodes


def solve_pin_flow(
    problem: RoutingProblem,
    *,
    edge_cost_fn,
    direction_fn,
    bend_penalty,
    overlap_block_weight,
):
    """Solve named pin-routing requests on a shared min-cost-flow network."""
    if not problem.requests:
        return SolverResult()

    route_names = tuple(request.key for request in problem.requests)
    target_specs = {
        request.key: request.target_candidates
        for request in problem.requests
    }
    start_nodes = {
        request.key: request.source_nodes
        for request in problem.requests
    }
    network = build_pin_min_cost_flow_network(
        route_names,
        target_specs,
        start_nodes,
        problem.graph.adj,
        lambda u, v, distance: edge_cost_fn(
            problem.edge_weights, u, v, distance
        ),
        lambda u, v: direction_fn(problem.graph, u, v),
        bend_penalty,
        overlap_block_weight,
    )
    if network is None:
        return SolverResult(
            failure=SolverFailure("no_targets", "No eligible pin targets"),
            objective_cost=float("inf"),
        )

    graph, source, sink, route_nodes = network
    flow, cost = min_cost_flow(graph, source, sink, len(route_names))
    if flow < len(route_names):
        return SolverResult(
            failure=SolverFailure(
                "insufficient_flow",
                f"Routed {flow} of {len(route_names)} requests",
            ),
            objective_cost=cost,
            completed_request_count=flow,
        )

    solved_routes = []
    route_node_count = 0
    for request in problem.requests:
        path, target = trace_flow_path(graph, route_nodes[request.key], sink)
        if path is None:
            return SolverResult(
                failure=SolverFailure(
                    "invalid_flow_path",
                    "Flow could not be reconstructed",
                    request.key,
                ),
                objective_cost=cost,
            )
        route_node_count += len(path)
        solved_routes.append(SolvedRoute(
            request.key,
            PinFlowRoute(tuple(path), target),
        ))

    return SolverResult(
        routes=tuple(solved_routes),
        route_node_count=route_node_count,
        objective_cost=cost,
        completed_request_count=flow,
    )
