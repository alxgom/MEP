"""Installation-neutral negotiated-congestion routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class NegotiatedProblem:
    """Network topology and dynamic target eligibility for one solve."""

    network_names: tuple[str, ...]
    start_nodes: Callable[[str], Sequence[int]]
    eligible_pins: Callable[[str, Mapping[str, str]], Sequence[str]]
    terminal_nodes: Mapping[str, int]
    congestion_scale: Callable[[str], float]


@dataclass
class NegotiatedRuntime:
    """Graph and route operations consumed by negotiated congestion."""

    env: object
    set_terminal_block_weight: Callable
    add_route_clearance_weights: Callable
    add_route_interaction_weights: Callable
    route_diameter: Callable
    route_segments_from_path: Callable
    route_axis_records: Callable
    run_search: Callable
    count_crossings: Callable
    score_routes: Callable


@dataclass(frozen=True)
class NegotiatedResult:
    success: bool
    routes: object
    total_nodes: int
    attempts: int
    crossing_free: bool


def solve_negotiated(
    problem: NegotiatedProblem,
    runtime: NegotiatedRuntime,
    pin_node_map,
    global_pins,
    *,
    machine_angle,
    bend_cost,
    iterations=20,
    present_penalty=20_000.0,
    history_penalty=4_000.0,
) -> NegotiatedResult:
    """Rip up and reroute networks until a crossing-free candidate is found."""
    current_paths = {}
    current_pins = {}
    current_pin_targets = {}
    history_congestion = {}
    node_history_congestion = {}
    best_routes = None
    best_score = float("inf")
    best_total_nodes = 0

    for attempt in range(1, iterations + 1):
        for network_name in problem.network_names:
            start_nodes = problem.start_nodes(network_name)
            if not start_nodes:
                continue
            target_pins = list(problem.eligible_pins(network_name, current_pins))

            current_paths[network_name] = None
            edge_usage, node_usage = _path_usage(current_paths.values())
            edge_weights = _weights_for_network(
                network_name,
                current_paths,
                problem,
                runtime,
                edge_usage,
                node_usage,
                history_congestion,
                node_history_congestion,
                present_penalty,
            )
            path, _, chosen_pin, chosen_target = runtime.run_search(
                runtime.env,
                start_nodes,
                target_pins,
                pin_node_map,
                global_pins,
                machine_angle,
                bend_cost,
                edge_weights=edge_weights,
            )
            if path is not None:
                current_paths[network_name] = path
                current_pins[network_name] = chosen_pin
                current_pin_targets[network_name] = chosen_target

        routes, total_nodes = _build_routes(
            problem.network_names,
            current_paths,
            current_pins,
            current_pin_targets,
            global_pins,
            runtime,
        )
        if routes is None:
            continue

        crossings = runtime.count_crossings(routes)
        score = runtime.score_routes(routes, crossings)
        if score < best_score:
            best_score = score
            best_routes = routes
            best_total_nodes = total_nodes
        if crossings == 0:
            return NegotiatedResult(True, routes, total_nodes, attempt, True)

        _update_history(
            current_paths.values(),
            history_congestion,
            node_history_congestion,
            history_penalty,
        )

    if best_routes is None:
        return NegotiatedResult(False, None, 0, iterations, False)
    return NegotiatedResult(True, best_routes, best_total_nodes, iterations, False)


def _weights_for_network(
    network_name,
    current_paths,
    problem,
    runtime,
    edge_usage,
    node_usage,
    history_congestion,
    node_history_congestion,
    present_penalty,
):
    weights = {}
    for terminal_network, terminal_node in problem.terminal_nodes.items():
        if terminal_network == network_name or terminal_node not in runtime.env.adj:
            continue
        for neighbour, _, _ in runtime.env.adj[terminal_node]:
            runtime.set_terminal_block_weight(weights, terminal_node, neighbour)

    scale = float(problem.congestion_scale(network_name))
    for node, neighbours in runtime.env.adj.items():
        for neighbour, distance, _ in neighbours:
            edge = (min(node, neighbour), max(node, neighbour))
            if weights.get(edge, 0.0) >= 1e9:
                continue
            congestion = (
                edge_usage.get(edge, 0) * present_penalty
                + history_congestion.get(edge, 0.0)
                + max(node_usage.get(node, 0), node_usage.get(neighbour, 0)) * present_penalty
                + max(
                    node_history_congestion.get(node, 0.0),
                    node_history_congestion.get(neighbour, 0.0),
                )
            )
            weights[edge] = distance + congestion * scale

    runtime.add_route_clearance_weights(weights, network_name, runtime.env)
    prior_axes = []
    for other_name, other_path in current_paths.items():
        if other_path is None or other_name == network_name:
            continue
        other_segments = runtime.route_segments_from_path(other_name, other_path)
        prior_axes.extend(runtime.route_axis_records(other_name, other_segments))
    runtime.add_route_interaction_weights(
        prior_axes,
        runtime.route_diameter(network_name),
        weights,
        runtime.env,
    )
    return weights


def _build_routes(network_names, paths, pins, targets, global_pins, runtime):
    routes = []
    total_nodes = 0
    for network_name in network_names:
        path = paths.get(network_name)
        if path is None:
            return None, 0
        segments = runtime.route_segments_from_path(
            network_name,
            path,
            pins.get(network_name),
            global_pins,
            targets.get(network_name),
        )
        routes.append((network_name, segments))
        total_nodes += len(path)
    return routes, total_nodes


def _path_usage(paths):
    edge_usage, node_usage = {}, {}
    for path in paths:
        if path is None:
            continue
        for node in path:
            node_usage[node] = node_usage.get(node, 0) + 1
        for index in range(len(path) - 1):
            edge = (min(path[index], path[index + 1]), max(path[index], path[index + 1]))
            edge_usage[edge] = edge_usage.get(edge, 0) + 1
    return edge_usage, node_usage


def _update_history(paths, edge_history, node_history, penalty):
    edge_usage, node_usage = _path_usage(paths)
    for edge, count in edge_usage.items():
        if count > 1:
            edge_history[edge] = edge_history.get(edge, 0.0) + penalty
    for node, count in node_usage.items():
        if count > 1:
            node_history[node] = node_history.get(node, 0.0) + penalty
