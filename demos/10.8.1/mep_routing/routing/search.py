from __future__ import annotations

import math
import heapq
from numbers import Integral

from .clearance import weighted_edge_cost


def line_graph_dir_from_points(env, u, v):
    pu = env.nodes[u]
    pv = env.nodes[v]
    dx = float(pv[0] - pu[0])
    dy = float(pv[1] - pu[1])
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    return "N" if dy > 0 else "S"


def path_physical_length(env, path):
    return float(sum(
        math.hypot(
            env.nodes[path[i + 1]][0] - env.nodes[path[i]][0],
            env.nodes[path[i + 1]][1] - env.nodes[path[i]][1],
        )
        for i in range(len(path) - 1)
    ))


def target_heuristic(
    env,
    node_idx,
    incoming_dir,
    target_specs,
    bend_cost,
    heuristic_mode,
    machine_center,
    estimate_turns_fn,
):
    if node_idx < 0 or node_idx >= len(env.nodes):
        return 0.0
    if heuristic_mode == 3:
        return 0.0

    p = env.nodes[node_idx]
    if heuristic_mode == 2:
        cx, cy = float(machine_center[0]), float(machine_center[1])
        radius = 0.0
        for target in target_specs:
            t = env.nodes[int(target["node_idx"])]
            radius = max(radius, abs(float(t[0] - cx)) + abs(float(t[1] - cy)))
        return max(0.0, abs(float(p[0] - cx)) + abs(float(p[1] - cy)) - radius)

    best = float("inf")
    for target in target_specs:
        t = env.nodes[int(target["node_idx"])]
        h = abs(float(t[0] - p[0])) + abs(float(t[1] - p[1]))
        if heuristic_mode == 0:
            h += bend_cost * estimate_turns_fn(p, incoming_dir, t)
        if h < best:
            best = h
    return 0.0 if best == float("inf") else float(best)


def terminal_node_indices(terminals, shaft_node_idx, kd):
    terminal_nodes = {"Shaft": shaft_node_idx}
    for name, pt in terminals.items():
        _, node_idx = kd.query(pt)
        terminal_nodes[name] = int(node_idx)
    return terminal_nodes


def ordered_small_room_names(terminals, machine_center, room_keywords=("Bathroom", "Toilet", "Washroom"), excluded=("Kitchen",)):
    cx, cy = machine_center
    return sorted(
        [
            name
            for name in terminals.keys()
            if name not in excluded and any(keyword in name for keyword in room_keywords)
        ],
        key=lambda name: math.hypot(terminals[name][0] - cx, terminals[name][1] - cy),
    )


def _start_node_list(start_node_indices):
    if isinstance(start_node_indices, Integral):
        return [int(start_node_indices)]
    return list(start_node_indices or [])


def _target_specs(target_pin_names, pin_node_map):
    return [
        target
        for pin_name in target_pin_names
        for target in pin_node_map.get(pin_name, [])
    ]


def run_super_sink_state_astar(
    env,
    start_node_indices,
    target_pin_names,
    pin_node_map,
    bend_cost,
    edge_weights=None,
    heuristic_mode=0,
    machine_center=(0.0, 0.0),
    estimate_turns_fn=None,
):
    """Route from any start node to a directed pin target using state-expanded A*."""
    start_nodes = _start_node_list(start_node_indices)
    estimate_turns_fn = estimate_turns_fn or (lambda *_args: 0)
    if not target_pin_names or not start_nodes:
        return None, 0.0, None, None

    num_nodes = len(env.nodes)
    super_source_idx = num_nodes
    super_sink_idx = num_nodes + 1
    target_specs = _target_specs(target_pin_names, pin_node_map)
    if not target_specs:
        return None, 0.0, None, None

    search_adj = {node_idx: list(neighbors) for node_idx, neighbors in env.adj.items()}
    search_adj[super_source_idx] = [(int(start_node), 0.0, None) for start_node in start_nodes]
    search_adj[super_sink_idx] = []

    pin_target_by_entry = {}
    for target in target_specs:
        pin_idx = int(target["node_idx"])
        pin_target_by_entry[(pin_idx, target["in_dir"])] = target
        search_adj.setdefault(pin_idx, []).append((super_sink_idx, 0.0, target["in_dir"]))
        search_adj[super_sink_idx].append((pin_idx, 0.0, target["out_dir"]))

    queue = []
    counter = 0
    g_scores = {(super_source_idx, None): 0.0}
    came_from = {}
    visited = set()
    heapq.heappush(queue, (0.0, 0.0, counter, super_source_idx, None))
    best_target_state = None

    while queue:
        _, current_cost, _, node_idx, incoming_dir = heapq.heappop(queue)
        state = (node_idx, incoming_dir)
        if state in visited:
            continue
        visited.add(state)
        if node_idx == super_sink_idx:
            best_target_state = state
            break

        for next_node, distance, edge_dir in search_adj.get(node_idx, []):
            next_node = int(next_node)
            edge_cost = weighted_edge_cost(edge_weights, node_idx, next_node, distance)
            turn_penalty = bend_cost if incoming_dir is not None and edge_dir is not None and incoming_dir != edge_dir else 0.0
            next_cost = current_cost + edge_cost + turn_penalty
            next_state = (next_node, edge_dir)
            if next_cost >= g_scores.get(next_state, float("inf")):
                continue
            g_scores[next_state] = next_cost
            came_from[next_state] = state
            heuristic = 0.0 if next_node >= num_nodes else target_heuristic(
                env,
                next_node,
                edge_dir,
                target_specs,
                bend_cost,
                heuristic_mode,
                machine_center,
                estimate_turns_fn,
            )
            counter += 1
            heapq.heappush(queue, (next_cost + heuristic, next_cost, counter, next_node, edge_dir))

    if best_target_state is None:
        return None, 0.0, None, None

    states = []
    current = best_target_state
    while current in came_from:
        states.append(current)
        current = came_from[current]
    states.append(current)
    states.reverse()

    path = [state[0] for state in states]
    if len(path) < 3:
        return None, 0.0, None, None

    chosen_pin_idx = path[-2]
    chosen_target = pin_target_by_entry.get((chosen_pin_idx, best_target_state[1]))
    chosen_pin_name = chosen_target["pin"] if chosen_target else target_pin_names[0]
    path_without_virtual = path[1:-1]
    return path_without_virtual, path_physical_length(env, path_without_virtual), chosen_pin_name, chosen_target


def run_super_sink_line_graph_search(
    env,
    start_node_indices,
    target_pin_names,
    pin_node_map,
    bend_cost,
    edge_weights=None,
    greedy=False,
    heuristic_mode=0,
    machine_center=(0.0, 0.0),
    estimate_turns_fn=None,
):
    """Route on directed graph edges, optionally using greedy best-first search."""
    start_nodes = _start_node_list(start_node_indices)
    estimate_turns_fn = estimate_turns_fn or (lambda *_args: 0)
    if not target_pin_names or not start_nodes:
        return None, 0.0, None, None

    target_specs = _target_specs(target_pin_names, pin_node_map)
    if not target_specs:
        return None, 0.0, None, None

    targets_by_node = {}
    for target in target_specs:
        targets_by_node.setdefault(int(target["node_idx"]), []).append(target)

    queue = []
    counter = 0
    g_scores = {}
    came_from = {}
    state_dirs = {}
    for start_node in start_nodes:
        for next_node, distance, edge_dir in env.adj.get(int(start_node), []):
            cost = weighted_edge_cost(edge_weights, int(start_node), int(next_node), distance)
            state = (int(start_node), int(next_node))
            if cost >= g_scores.get(state, float("inf")):
                continue
            g_scores[state] = cost
            state_dir = edge_dir if edge_dir is not None else line_graph_dir_from_points(env, int(start_node), int(next_node))
            state_dirs[state] = state_dir
            heuristic = target_heuristic(
                env,
                int(next_node),
                state_dir,
                target_specs,
                bend_cost,
                heuristic_mode,
                machine_center,
                estimate_turns_fn,
            )
            priority = heuristic if greedy else cost + heuristic
            heapq.heappush(queue, (priority, cost, counter, state))
            counter += 1

    best_final_cost = float("inf")
    best_final_state = None
    best_target = None
    visited = set()
    while queue:
        f_score, current_cost, _, state = heapq.heappop(queue)
        if not greedy and f_score >= best_final_cost:
            break
        if state in visited:
            continue
        visited.add(state)

        previous_node, node_idx = state
        current_dir = state_dirs[state]
        for target in targets_by_node.get(node_idx, []):
            final_penalty = bend_cost if current_dir != target["in_dir"] else 0.0
            final_cost = current_cost + final_penalty
            if final_cost < best_final_cost:
                best_final_cost = final_cost
                best_final_state = state
                best_target = target
        if greedy and best_final_state is not None:
            break

        for next_node, distance, next_dir in env.adj.get(node_idx, []):
            next_node = int(next_node)
            if next_node == previous_node:
                continue
            edge_cost = weighted_edge_cost(edge_weights, node_idx, next_node, distance)
            turn_penalty = bend_cost if current_dir != next_dir else 0.0
            next_state = (node_idx, next_node)
            next_cost = current_cost + edge_cost + turn_penalty
            if next_cost >= g_scores.get(next_state, float("inf")):
                continue
            g_scores[next_state] = next_cost
            came_from[next_state] = state
            state_dirs[next_state] = next_dir
            heuristic = target_heuristic(
                env,
                next_node,
                next_dir,
                target_specs,
                bend_cost,
                heuristic_mode,
                machine_center,
                estimate_turns_fn,
            )
            priority = heuristic if greedy else next_cost + heuristic
            heapq.heappush(queue, (priority, next_cost, counter, next_state))
            counter += 1

    if best_final_state is None or best_target is None:
        return None, 0.0, None, None

    states = []
    current = best_final_state
    while current in came_from:
        states.append(current)
        current = came_from[current]
    states.append(current)
    states.reverse()
    path = [states[0][0]]
    path.extend(state[1] for state in states)
    return path, path_physical_length(env, path), best_target["pin"], best_target
