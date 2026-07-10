from __future__ import annotations

import math


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
