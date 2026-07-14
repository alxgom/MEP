"""Sal topology adapter for shared negotiated-congestion routing."""

from dataclasses import dataclass

from mep_routing.routing.negotiated import (
    NegotiatedProblem,
    NegotiatedResult,
    NegotiatedRuntime,
    solve_negotiated,
)

from .policy import SalSolverPolicy
from .route_plan import SalRoutePlan


# Retain the established Sal-only topology lookups beside the shared runtime.
@dataclass
class SalNegotiatedContext(NegotiatedRuntime):
    route_start_nodes: object
    terminal_node_indices: object


NegotiatedRoutingResult = NegotiatedResult


def run_negotiated_congestion(
    room_names,
    pin_node_map,
    global_pins,
    shaft_boundary_nodes,
    shaft_node_idx,
    *,
    route_plan: SalRoutePlan,
    policy: SalSolverPolicy,
    context: SalNegotiatedContext,
    machine_angle,
    favour_large=False,
):
    """Apply Sal network starts, port eligibility, and tuning to the shared solver."""

    def start_nodes(route_name):
        if route_name == route_plan.shaft_route:
            return shaft_boundary_nodes
        return context.route_start_nodes(route_name)

    def eligible_pins(route_name, current_pins):
        if route_name == route_plan.shaft_route:
            return route_plan.large_ports
        if route_name == route_plan.kitchen_route:
            shaft_pin = current_pins.get(
                route_plan.shaft_route,
                route_plan.large_ports[0],
            )
            return (route_plan.kitchen_port_for(shaft_pin),)

        used_small_pins = {
            current_pins[name]
            for name in room_names
            if name != route_name and name in current_pins
        }
        available = tuple(
            pin for pin in route_plan.small_ports if pin not in used_small_pins
        )
        return available or (route_plan.small_ports[0],)

    problem = NegotiatedProblem(
        network_names=route_plan.all_routes,
        start_nodes=start_nodes,
        eligible_pins=eligible_pins,
        terminal_nodes=context.terminal_node_indices(pin_node_map, shaft_node_idx),
        congestion_scale=lambda route_name: (
            policy.negotiated_large_route_factor
            if favour_large and route_name in route_plan.large_routes
            else 1.0
        ),
    )
    return solve_negotiated(
        problem,
        context,
        pin_node_map,
        global_pins,
        machine_angle=machine_angle,
        bend_cost=policy.bend_cost,
        iterations=policy.negotiated_iterations,
        present_penalty=policy.negotiated_present_penalty,
        history_penalty=policy.negotiated_history_penalty,
    )
