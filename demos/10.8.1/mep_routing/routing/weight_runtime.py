"""Runtime coordination for routing edge-clearance weights and overlays."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from .clearance import (
    add_machine_clearance_weights as _add_machine_clearance_weights,
    add_route_interaction_weights as _add_route_interaction_weights,
    add_static_clearance_weights as _add_static_clearance_weights,
    route_axis_records,
    static_clearance_cache_key,
    static_clearance_distances,
    static_shaft_distance_segments,
    static_wall_distance_segments,
    set_block_weight,
)
from .search import run_super_sink_search


@dataclass
class StaticClearanceCache:
    """Cached graph-edge distance fields for static routing geometry."""

    key: Any = None
    wall_distances: Any = None
    shaft_distances: Any = None


@dataclass
class EdgeWeightOverlay:
    """Display-friendly view of non-baseline routing edge weights."""

    values: dict[tuple[int, int], float] = field(default_factory=dict)
    excluded_edges: set[tuple[int, int]] = field(default_factory=set)

    def reset(self) -> None:
        self.values.clear()
        self.excluded_edges.clear()


@dataclass(frozen=True)
class RoutingWeightRuntimeContext:
    """Live graph, geometry, and policy dependencies for edge-weight assembly."""

    edge_list: Sequence[Sequence[Any]]
    edge_coords: Any
    nodes: Any
    routing_region: Any
    room_polygons: Sequence[Any]
    walls: Sequence[Any]
    wall_polygons: Sequence[Any]
    shafts: Sequence[Any]
    machine_center: tuple[float, float]
    machine_angle_deg: float
    machine_overall_width_mm: float
    machine_body_height_mm: float
    buffer_ratio: float
    shaft_clearance_mm: float
    machine_soft_margin_mm: float
    crossing_penalty: float
    clearance_penalty: float
    block_weight: float
    route_diameter: Callable[[str], float]


class RoutingRuntime:
    """Concrete weight and search service for one live routing graph."""

    def __init__(
        self,
        env,
        context: RoutingWeightRuntimeContext,
        cache: StaticClearanceCache,
        overlay: EdgeWeightOverlay,
        *,
        search_backend,
        heuristic_mode: int,
        bend_cost: float,
        estimate_turns_fn: Callable,
    ) -> None:
        self.env = env
        self.context = context
        self.cache = cache
        self.overlay = overlay
        self.search_backend = search_backend
        self.heuristic_mode = heuristic_mode
        self.bend_cost = bend_cost
        self.estimate_turns_fn = estimate_turns_fn

    def route_axis_records(self, route_name, route_segments):
        return route_axis_records_for_routes(route_name, route_segments, self.context)

    def add_static_clearance_weights(self, edge_weights, route_diameter, *, allow_shaft_entry=False):
        return add_static_clearance_weights(
            edge_weights,
            route_diameter,
            self.context,
            self.cache,
            allow_shaft_entry=allow_shaft_entry,
        )

    def add_machine_clearance_weights(self, edge_weights, route_diameter):
        return add_machine_clearance_weights(edge_weights, route_diameter, self.context)

    def add_route_clearance_weights(self, edge_weights, route_name, *, shaft_route_name="Shaft"):
        return add_route_clearance_weights(
            edge_weights,
            route_name,
            self.context,
            self.cache,
            shaft_route_name=shaft_route_name,
        )

    def add_route_interaction_weights(self, prior_axes, route_diameter, edge_weights):
        return add_route_interaction_weights(
            prior_axes,
            route_diameter,
            edge_weights,
            self.context,
        )

    def set_terminal_block_weight(self, edge_weights, u, v):
        edge = set_block_weight(edge_weights, u, v, self.context.block_weight)
        self.overlay.excluded_edges.add(edge)
        return edge

    def record_edge_weight_overlay(self, edge_weights):
        if edge_weights and self.env is not None:
            record_weight_overlay(edge_weights, self.context, self.overlay)

    def refresh_edge_weight_overlay(self, routes, route_diameter):
        return refresh_weight_overlay(
            routes,
            route_diameter,
            self.context,
            self.cache,
            self.overlay,
        )

    def run_super_sink_search(
        self,
        start_node_indices,
        target_pin_names,
        pin_node_map,
        *,
        edge_weights=None,
        bend_cost=None,
    ):
        self.record_edge_weight_overlay(edge_weights)
        return run_super_sink_search(
            self.search_backend,
            self.env,
            start_node_indices,
            target_pin_names,
            pin_node_map,
            self.bend_cost if bend_cost is None else bend_cost,
            edge_weights=edge_weights,
            heuristic_mode=self.heuristic_mode,
            machine_center=self.context.machine_center,
            estimate_turns_fn=self.estimate_turns_fn,
        )


def static_clearance_fields(
    context: RoutingWeightRuntimeContext,
    cache: StaticClearanceCache,
):
    """Return cached wall and shaft distance fields for the active graph."""
    if context.edge_coords is None or len(context.edge_coords) == 0:
        return None, None

    key = static_clearance_cache_key(
        context.routing_region,
        context.edge_list,
        context.room_polygons,
        context.wall_polygons,
        context.shafts,
    )
    if cache.key == key:
        return cache.wall_distances, cache.shaft_distances

    wall_distances, shaft_distances = static_clearance_distances(
        context.edge_coords,
        static_wall_distance_segments(
            context.routing_region,
            context.room_polygons,
            context.walls,
            context.wall_polygons,
        ),
        static_shaft_distance_segments(context.shafts),
    )
    cache.key = key
    cache.wall_distances = wall_distances
    cache.shaft_distances = shaft_distances
    return wall_distances, shaft_distances


def route_axis_records_for_routes(
    route_name: str,
    route_segments,
    context: RoutingWeightRuntimeContext,
):
    return route_axis_records(route_name, route_segments, context.route_diameter)


def add_static_clearance_weights(
    edge_weights: MutableMapping[tuple[int, int], float],
    route_diameter_mm: float,
    context: RoutingWeightRuntimeContext,
    cache: StaticClearanceCache,
    *,
    allow_shaft_entry: bool = False,
) -> None:
    wall_distances, shaft_distances = static_clearance_fields(context, cache)
    _add_static_clearance_weights(
        edge_weights,
        context.edge_list,
        wall_distances,
        shaft_distances,
        route_diameter_mm,
        context.buffer_ratio,
        context.shaft_clearance_mm,
        context.block_weight,
        allow_shaft_entry=allow_shaft_entry,
    )


def add_machine_clearance_weights(
    edge_weights: MutableMapping[tuple[int, int], float],
    route_diameter_mm: float,
    context: RoutingWeightRuntimeContext,
) -> None:
    _add_machine_clearance_weights(
        edge_weights,
        context.edge_list,
        context.edge_coords,
        context.nodes,
        route_diameter_mm,
        context.buffer_ratio,
        machine_center=context.machine_center,
        machine_angle_deg=context.machine_angle_deg,
        machine_overall_width_mm=context.machine_overall_width_mm,
        machine_body_height_mm=context.machine_body_height_mm,
        soft_margin_mm=context.machine_soft_margin_mm,
        clearance_penalty=context.clearance_penalty,
        block_weight=context.block_weight,
    )


def add_route_clearance_weights(
    edge_weights: MutableMapping[tuple[int, int], float],
    route_name: str,
    context: RoutingWeightRuntimeContext,
    cache: StaticClearanceCache,
    shaft_route_name: str = "Shaft",
) -> None:
    diameter = context.route_diameter(route_name)
    add_static_clearance_weights(
        edge_weights,
        diameter,
        context,
        cache,
        allow_shaft_entry=route_name == shaft_route_name,
    )
    add_machine_clearance_weights(edge_weights, diameter, context)


def add_route_interaction_weights(
    prior_axis_records,
    current_diameter_mm: float,
    edge_weights: MutableMapping[tuple[int, int], float],
    context: RoutingWeightRuntimeContext,
) -> None:
    _add_route_interaction_weights(
        prior_axis_records,
        current_diameter_mm,
        edge_weights,
        context.edge_list,
        context.edge_coords,
        context.nodes,
        context.buffer_ratio,
        context.crossing_penalty,
        context.clearance_penalty,
        context.block_weight,
    )


def record_weight_overlay(
    edge_weights: Mapping[tuple[int, int], float],
    context: RoutingWeightRuntimeContext,
    overlay: EdgeWeightOverlay,
) -> None:
    """Record blocks and penalties without exposing terminal-blocked edges."""
    for (u, v), cost in edge_weights.items():
        if u < 0 or v < 0 or u >= len(context.nodes) or v >= len(context.nodes):
            continue
        edge = (min(int(u), int(v)), max(int(u), int(v)))
        if edge in overlay.excluded_edges:
            continue
        base_length = math.hypot(
            float(context.nodes[v][0] - context.nodes[u][0]),
            float(context.nodes[v][1] - context.nodes[u][1]),
        )
        if base_length <= 1e-7:
            continue
        if cost >= context.block_weight:
            overlay.values[edge] = max(overlay.values.get(edge, 0.0), context.block_weight)
            continue
        added_cost = float(cost) - base_length
        if added_cost > 1e-7:
            overlay.values[edge] = max(overlay.values.get(edge, 0.0), added_cost / base_length)


def refresh_weight_overlay(
    routes,
    route_diameter_mm: float,
    context: RoutingWeightRuntimeContext,
    cache: StaticClearanceCache,
    overlay: EdgeWeightOverlay,
) -> EdgeWeightOverlay:
    """Rebuild the display overlay for the selected duct-diameter mode."""
    overlay.reset()
    edge_weights: dict[tuple[int, int], float] = {}
    add_static_clearance_weights(edge_weights, route_diameter_mm, context, cache)
    add_machine_clearance_weights(edge_weights, route_diameter_mm, context)

    prior_axis_records = []
    for route_name, route_segments in routes or ():
        prior_axis_records.extend(route_axis_records_for_routes(route_name, route_segments, context))
    add_route_interaction_weights(prior_axis_records, route_diameter_mm, edge_weights, context)
    record_weight_overlay(edge_weights, context, overlay)
    return overlay
