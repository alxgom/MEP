"""Cli minimum duct-envelope costs applied to reusable graph adjacency."""

from __future__ import annotations

from mep_routing.routing import (
    add_machine_clearance_weights,
    add_static_clearance_weights,
    static_clearance_distances,
    static_wall_distance_segments,
)


CLI_MINIMUM_DUCT_WIDTH_MM = 150
CLI_MINIMUM_CENTERLINE_CLEARANCE_MM = CLI_MINIMUM_DUCT_WIDTH_MM // 2
CLI_OBSTACLE_WEIGHT = 1_000_000.0


def cli_minimum_clearance_weights(
    prepared,
    machine,
    center,
    angle_degrees,
    runtime,
    *,
    clearance_mm=CLI_MINIMUM_CENTERLINE_CLEARANCE_MM,
    obstacle_weight=CLI_OBSTACLE_WEIGHT,
):
    """Return near-hard costs for Cli edges too close to walls or its machine."""
    wall_segments = static_wall_distance_segments(
        prepared.routing_region_base,
        [room.polygon for room in prepared.rooms],
        prepared.walls,
        prepared.wall_polygons,
    )
    wall_distances, _ = static_clearance_distances(
        runtime.edge_coords, wall_segments, (),
    )
    weights = {}
    add_static_clearance_weights(
        weights,
        runtime.edge_list,
        wall_distances,
        None,
        route_diameter_mm=int(clearance_mm) * 2,
        buffer_ratio=1.0,
        shaft_clearance_mm=0,
        block_weight=float(obstacle_weight),
    )
    body_width, body_depth, _body_height = machine.body_size_mm
    add_machine_clearance_weights(
        weights,
        runtime.edge_list,
        runtime.edge_coords,
        runtime.nodes,
        route_diameter_mm=int(clearance_mm) * 2,
        buffer_ratio=1.0,
        machine_center=center,
        machine_angle_deg=angle_degrees,
        machine_overall_width_mm=body_width,
        machine_body_height_mm=body_depth,
        soft_margin_mm=0,
        clearance_penalty=0,
        block_weight=float(obstacle_weight),
    )
    return weights


def apply_cli_edge_weights(adjacency, weights):
    """Copy adjacency while replacing baseline distances with Cli edge costs."""
    return {
        int(node): [
            (
                int(neighbor),
                float(weights.get((min(int(node), int(neighbor)), max(int(node), int(neighbor))), distance)),
                direction,
            )
            for neighbor, distance, direction in neighbors
        ]
        for node, neighbors in adjacency.items()
    }
