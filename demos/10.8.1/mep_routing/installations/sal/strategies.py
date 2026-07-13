"""Sal-specific orchestration policies composed from shared routing primitives."""

from .routes import KITCHEN_ROUTE_NAME, SHAFT_ROUTE_NAME


def run_sequential_routing(
    small_route_order,
    pin_node_map,
    global_pins,
    shaft_node_idx,
    chosen_exhaust_pin,
    chosen_exhaust_target,
    shaft_path,
    *,
    env,
    machine_angle,
    bend_cost,
    route_start_nodes,
    route_segments_from_path,
    run_search,
    terminal_node_indices,
    set_terminal_block_weight,
    add_route_clearance_weights,
    add_route_interaction_weights,
    route_diameter,
    route_axis_records,
):
    """Route Sal's shaft, kitchen, then small ducts in the supplied order."""
    base_weights = {}
    prior_axis_records = []
    kitchen_pin_name = "right_mid" if chosen_exhaust_pin == "left_mid" else "left_mid"
    routes = []

    shaft_segs = route_segments_from_path(
        SHAFT_ROUTE_NAME,
        shaft_path,
        chosen_exhaust_pin,
        global_pins,
        chosen_exhaust_target,
    )
    routes.append((SHAFT_ROUTE_NAME, shaft_segs))
    prior_axis_records.extend(route_axis_records(SHAFT_ROUTE_NAME, shaft_segs))
    total_nodes = len(shaft_path)
    terminal_nodes = terminal_node_indices(pin_node_map, shaft_node_idx)

    def weights_for_route(route_name):
        weights = base_weights.copy()
        for terminal_route, terminal_node_idx in terminal_nodes.items():
            if terminal_route == route_name:
                continue
            if terminal_node_idx in env.adj:
                for neighbour, _, _ in env.adj[terminal_node_idx]:
                    set_terminal_block_weight(weights, terminal_node_idx, neighbour)
        add_route_clearance_weights(weights, route_name, env)
        add_route_interaction_weights(
            prior_axis_records,
            route_diameter(route_name),
            weights,
            env,
        )
        return weights

    kitchen_start_nodes = route_start_nodes(KITCHEN_ROUTE_NAME)
    if kitchen_start_nodes:
        kitchen_path, _, _, kitchen_target = run_search(
            env,
            kitchen_start_nodes,
            [kitchen_pin_name],
            pin_node_map,
            global_pins,
            machine_angle,
            bend_cost,
            edge_weights=weights_for_route(KITCHEN_ROUTE_NAME),
        )
        if kitchen_path is None:
            return False, None, "No path to Kitchen", 0

        kitchen_segs = route_segments_from_path(
            KITCHEN_ROUTE_NAME,
            kitchen_path,
            kitchen_pin_name,
            global_pins,
            kitchen_target,
        )
        routes.append((KITCHEN_ROUTE_NAME, kitchen_segs))
        prior_axis_records.extend(route_axis_records(KITCHEN_ROUTE_NAME, kitchen_segs))
        total_nodes += len(kitchen_path)

    available_small_pins = ["tl", "tr", "bl", "br"]
    for route_name in small_route_order:
        if not available_small_pins:
            return False, None, f"No port for {route_name}", 0
        route_start = route_start_nodes(route_name)
        if not route_start:
            return False, None, f"No start nodes for {route_name}", 0

        route_path, _, chosen_small_pin, route_target = run_search(
            env,
            route_start,
            available_small_pins,
            pin_node_map,
            global_pins,
            machine_angle,
            bend_cost,
            edge_weights=weights_for_route(route_name),
        )
        if route_path is None:
            return False, None, f"No path to {route_name}", 0

        route_segs = route_segments_from_path(
            route_name,
            route_path,
            chosen_small_pin,
            global_pins,
            route_target,
        )
        routes.append((route_name, route_segs))
        prior_axis_records.extend(route_axis_records(route_name, route_segs))
        total_nodes += len(route_path)
        available_small_pins.remove(chosen_small_pin)

    return True, routes, "Success", total_nodes


def select_two_stage_routing(big_first_runner, small_first_runner, count_crossings, score_routes):
    """Choose Sal's lower-scoring complete large/small routing stage order."""
    candidates = []
    for runner in (big_first_runner, small_first_runner):
        success, routes, status, total_nodes = runner()
        if not success:
            continue
        candidates.append((score_routes(routes, count_crossings(routes)), routes, total_nodes, status))

    if not candidates:
        return False, None, "Two-stage min-cost flow found no complete stage order", 0

    _, best_routes, best_total_nodes, best_status = min(candidates, key=lambda item: item[0])
    return True, best_routes, best_status, best_total_nodes


def search_large_route_candidates(
    pin_node_map, shaft_boundary_nodes, *, env, terminals, route_start_nodes,
    route_one_pin_flow, route_segments_from_path, route_axis_records,
    add_route_clearance_weights, add_route_interaction_weights, route_diameter,
    count_crossings, score_routes, initial_edge_weights=None,
):
    """Evaluate Sal's large-duct port assignments and routing orders."""
    if not shaft_boundary_nodes or KITCHEN_ROUTE_NAME not in terminals:
        return None, None, float("inf"), 0, {}
    kitchen_starts = route_start_nodes(KITCHEN_ROUTE_NAME)
    if not kitchen_starts:
        return None, None, float("inf"), 0, {}

    best = (None, None, float("inf"), 0, {})
    assignments = (
        {SHAFT_ROUTE_NAME: "left_mid", KITCHEN_ROUTE_NAME: "right_mid"},
        {SHAFT_ROUTE_NAME: "right_mid", KITCHEN_ROUTE_NAME: "left_mid"},
    )
    for assignment in assignments:
        for shaft_start in shaft_boundary_nodes[:1]:
            terminal_points = {SHAFT_ROUTE_NAME: env.nodes[int(shaft_start)], KITCHEN_ROUTE_NAME: kitchen_starts}
            for order in ((SHAFT_ROUTE_NAME, KITCHEN_ROUTE_NAME), (KITCHEN_ROUTE_NAME, SHAFT_ROUTE_NAME)):
                paths, targets, prior_axes, total_cost = {}, {}, [], 0.0
                for route_name in order:
                    weights = (initial_edge_weights or {}).copy()
                    add_route_clearance_weights(weights, route_name, env)
                    add_route_interaction_weights(prior_axes, route_diameter(route_name), weights, env)
                    path, target, cost = route_one_pin_flow(route_name, assignment[route_name], terminal_points[route_name], pin_node_map, edge_weights=weights)
                    if path is None:
                        break
                    paths[route_name], targets[route_name] = path, target
                    total_cost += cost
                    prior_axes.extend(route_axis_records(route_name, route_segments_from_path(route_name, path, target["pin"], None, target)))
                if len(paths) != 2:
                    continue
                routes = [(name, route_segments_from_path(name, paths[name], targets[name]["pin"], None, targets[name])) for name in (SHAFT_ROUTE_NAME, KITCHEN_ROUTE_NAME)]
                score = score_routes(routes, count_crossings(routes))
                if score < best[2]:
                    best = (paths, targets, score, 2, {"assignment": f"Shaft={assignment[SHAFT_ROUTE_NAME]},Kitchen={assignment[KITCHEN_ROUTE_NAME]}", "large_order": "->".join(order)})
    return best


def run_small_flow_stage(room_names, pin_node_map, global_pins, prior_axis_records, *, small_diameter, env, build_weights, run_flow, build_routes):
    """Run Sal's shared small-duct flow stage after the supplied prior routes."""
    weights = build_weights(prior_axis_records, small_diameter, env)
    paths, targets, _, flow = run_flow(room_names, pin_node_map, edge_weights=weights)
    if paths is None:
        return False, None, f"Min-cost flow routed {flow}/{len(room_names)} small ducts", 0
    routes, nodes = build_routes(room_names, paths, targets, global_pins)
    if routes is None:
        return False, None, "Could not build small duct routes", 0
    return True, routes, "Success", nodes
