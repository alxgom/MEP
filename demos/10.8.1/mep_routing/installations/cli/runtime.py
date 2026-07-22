"""Cli-owned interactive supply and return-air runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from mep_routing.app import InstallationGraphProfile, create_active_dwelling
from mep_routing.graphs import GraphMode

from .geometry import cli_air_access_spec, cli_machine_pins
from .clearance import apply_cli_edge_weights, cli_minimum_clearance_weights
from .duct_geometry import build_core_duct_polygon_projection
from .graph_inputs import build_cli_simple_graph_inputs
from .grilles import CliGrillePlan, prepare_cli_grilles
from .materializer import CliDuctSegment, materialize_cli_tree
from .placement import CliPlacementResult, CliPlacementRuntime
from .solver_backends import solve_cli_tree
from .sizing import (
    CliSizingContext, CoreConstantFrictionSizing, CoreSupplyGrilleSizing,
    size_tree_edges,
)
from .terminals import is_cli_supply_room


@dataclass(frozen=True)
class CliRoutingResult:
    routes: tuple[tuple[str, list], ...]
    status: str
    elapsed_ms: float
    total_nodes: int
    terminals: dict[str, tuple[int, int]]
    machine_pins: dict[str, tuple[int, int]]
    workspace: object | None = None
    grille_plan: CliGrillePlan | None = None
    supply_segments: tuple[CliDuctSegment, ...] = ()
    return_segments: tuple[CliDuctSegment, ...] = ()
    placement: CliPlacementResult | None = None
    context_obstacle_count: int = 0


@dataclass(frozen=True)
class _SolvedTree:
    workspace: object
    segments: tuple[CliDuctSegment, ...]
    node_count: int


class CliInteractiveRuntime:
    """Compose Cli capabilities without consuming Sal placement or solver contracts."""

    def __init__(
        self, definition, graph_config, terminal_config, placement=None,
        sizing_policy_factory=CoreConstantFrictionSizing,
        grille_sizing_policy_factory=CoreSupplyGrilleSizing,
    ):
        self.definition = definition
        self.graph_config = graph_config
        self.terminal_config = terminal_config
        self.placement = placement or CliPlacementRuntime()
        self.sizing_policy_factory = sizing_policy_factory
        self.grille_sizing_policy_factory = grille_sizing_policy_factory
        self._core_polygon_result = None
        self._core_polygon_projection = None
        self.last_placement = None
        self.last_result = CliRoutingResult((), "Cli has not run", 0.0, 0, {}, {})

    def place_machine(self, prepared_dwelling, machine, *, context_obstacles=()):
        grille_sizes = self._size_supply_grilles(prepared_dwelling.rooms, machine)
        grilles = prepare_cli_grilles(
            prepared_dwelling.rooms, prepared_dwelling.routing_region_base, machine=machine,
            supply_width_by_room_id=grille_sizes,
        )
        self.last_placement = self.placement.place(
            prepared_dwelling, machine, grilles, context_obstacles=context_obstacles,
        )
        return self.last_placement

    def solve(
        self, prepared_dwelling, machine, center, angle_degrees=0, *,
        context_obstacles=(), return_enabled=True, graph_mode=GraphMode.CORE_SIMPLE,
        solver_backend_index=0, bend_penalty_mm=4000.0,
        crossing_penalty_multiplier=5.0,
    ):
        started = perf_counter()
        pins = cli_machine_pins(machine, center, angle_degrees)
        grille_sizes = self._size_supply_grilles(prepared_dwelling.rooms, machine)
        grilles = prepare_cli_grilles(
            prepared_dwelling.rooms, prepared_dwelling.routing_region_base,
            machine=machine, machine_pins=pins, angle_degrees=angle_degrees,
            supply_width_by_room_id=grille_sizes,
        )
        if not return_enabled:
            grilles = CliGrillePlan(tuple(grilles.supply))
        if not grilles.supply:
            return self._finish(
                (), "Blocked: no Cli supply grilles", started, 0, grilles, pins,
                context_obstacles=context_obstacles,
            )

        supply = self._solve_tree(
            "supply", prepared_dwelling, machine, pins, angle_degrees,
            grilles, context_obstacles, graph_mode, solver_backend_index,
            bend_penalty_mm, crossing_penalty_multiplier,
        )
        if supply is None:
            if context_obstacles:
                relaxed = self.solve(
                    prepared_dwelling, machine, center, angle_degrees,
                    return_enabled=return_enabled, graph_mode=graph_mode,
                    solver_backend_index=solver_backend_index,
                    bend_penalty_mm=bend_penalty_mm,
                    crossing_penalty_multiplier=crossing_penalty_multiplier,
                )
                if relaxed.routes:
                    self.last_result = replace(
                        relaxed,
                        status=f"Cli context relaxed after collision; {relaxed.status}",
                        context_obstacle_count=len(tuple(context_obstacles)),
                    )
                    return self.last_result
            return self._finish(
                (), "Blocked: no Cli supply tree", started, 0, grilles, pins,
                context_obstacles=context_obstacles,
            )
        returned = None
        if return_enabled:
            returned = self._solve_tree(
                "return", prepared_dwelling, machine, pins, angle_degrees,
                grilles, context_obstacles, graph_mode, solver_backend_index,
                bend_penalty_mm, crossing_penalty_multiplier,
                prior_segments=tuple(segment.line for segment in supply.segments),
            )
        routes = (("Supply Air Tree", [segment.line for segment in supply.segments]),)
        status = f"Cli supply solved: {len(grilles.supply)} grilles"
        return_segments = ()
        total_nodes = supply.node_count
        if returned is not None:
            return_segments = returned.segments
            routes += (("Return Air Tree", [segment.line for segment in returned.segments]),)
            total_nodes += returned.node_count
            status = (
                f"Cli supply/return solved: {len(grilles.supply)} rooms, "
                f"{len(supply.segments) + len(return_segments)} segments"
            )
        elif return_enabled:
            status += "; return tree blocked"
        return self._finish(
            routes, status, started, total_nodes, grilles, pins,
            workspace=supply.workspace, supply_segments=supply.segments,
            return_segments=return_segments, context_obstacles=context_obstacles,
        )

    def _size_supply_grilles(self, rooms, machine):
        if not machine.airflow_m3h:
            return {}
        targets = tuple(room for room in rooms if is_cli_supply_room(room))
        areas = {id(room): float(room.polygon.area) / 1_000_000.0 for room in targets}
        total_area = sum(areas.values())
        if total_area <= 0:
            return {}
        policy = self.grille_sizing_policy_factory(machine.airflow_m3h)
        return {
            id(room): policy.size(areas[id(room)], total_area)
            for room in targets
        }

    def _solve_tree(
        self, role, prepared, machine, pins, angle, grilles, context_obstacles,
        graph_mode, solver_backend_index, bend_penalty_mm,
        crossing_penalty_multiplier, prior_segments=(),
    ):
        connections = grilles.supply if role == "supply" else grilles.returns
        if not connections:
            return None
        connector_key = "air_out" if role == "supply" else "air_in"
        access_length_mm = 500 if role == "supply" else 20
        access = cli_air_access_spec(
            machine, pins, angle, connector_key, access_length_mm,
        )
        terminals = {item.route_name: item.routing_point for item in connections}
        role_prepared = replace(prepared, terminals=terminals, wet_room_names=[])

        def simple_inputs(**kwargs):
            return build_cli_simple_graph_inputs(
                **kwargs, context_obstacles=context_obstacles,
            )

        profile = InstallationGraphProfile(
            port_access_specs=lambda current_pins, current_angle: (
                cli_air_access_spec(
                    machine, current_pins, current_angle, connector_key,
                    access_length_mm,
                ),
            ),
            simple_input_builder=simple_inputs,
        )
        dwelling = create_active_dwelling(
            role_prepared, self.graph_config, profile, self.terminal_config,
        )
        build = dwelling.workspace.build_selected(GraphMode(graph_mode), pins, angle)
        if build is None or not len(build.runtime.nodes):
            return None
        runtime = build.runtime
        corner_keys = ("c_tl", "c_tr", "c_br", "c_bl")
        machine_center = (
            sum(pins[key][0] for key in corner_keys) / len(corner_keys),
            sum(pins[key][1] for key in corner_keys) / len(corner_keys),
        )
        clearance_weights = cli_minimum_clearance_weights(
            prepared, machine, machine_center, angle, runtime,
        )
        weighted_adjacency = apply_cli_edge_weights(runtime.adjacency, clearance_weights)
        dwelling.workspace.overlay.values = dict(clearance_weights)
        points = [access["access_point"], *terminals.values()]
        indices = tuple(dict.fromkeys(int(runtime.spatial_index.query(point)[1]) for point in points))
        tree = solve_cli_tree(
            solver_backend_index, runtime.nodes, weighted_adjacency, indices,
            bend_penalty=bend_penalty_mm,
            crossing_multiplier=crossing_penalty_multiplier,
            prior_segments=prior_segments,
        )
        if tree is None:
            return None
        edge_sizes = {}
        machine_sizing = None
        grille_sizes = {}
        if role == "supply" and machine.airflow_m3h and machine.pressure_loss_mmca:
            area_by_route = {item.route_name: item.room_area_m2 for item in connections}
            area_by_node = {}
            for name, point in terminals.items():
                node = int(runtime.spatial_index.query(point)[1])
                area_by_node[node] = area_by_node.get(node, 0.0) + area_by_route.get(name, 0.0)
            policy = self.sizing_policy_factory(CliSizingContext(
                machine.airflow_m3h, machine.pressure_loss_mmca,
            ))
            total_area = float(sum(area_by_route.values()))
            if total_area > 0:
                machine_sizing = policy.size(total_area, total_area)
                grille_sizes = {
                    item.route_name: policy.size(item.room_area_m2, total_area)
                    for item in connections
                }
            root_node = int(runtime.spatial_index.query(access["access_point"])[1])
            edge_sizes = size_tree_edges(tree.tree_edges, root_node, area_by_node, policy)
        segments = materialize_cli_tree(
            role=role, machine=machine, pins=pins, access_spec=access,
            graph_nodes=runtime.nodes, tree=tree, grille_connections=connections,
            edge_sizes=edge_sizes, machine_sizing=machine_sizing,
            grille_sizes=grille_sizes,
        )
        return _SolvedTree(dwelling.workspace, segments, len(runtime.nodes))

    def _finish(
        self, routes, status, started, total_nodes, grilles, pins, *, workspace=None,
        supply_segments=(), return_segments=(), context_obstacles=(), placement=None,
    ):
        self.last_result = CliRoutingResult(
            tuple(routes), status, (perf_counter() - started) * 1000.0, total_nodes,
            grilles.grille_terminals(), dict(pins), workspace, grilles,
            tuple(supply_segments), tuple(return_segments), placement or self.last_placement,
            len(tuple(context_obstacles)),
        )
        self._core_polygon_result = None
        self._core_polygon_projection = None
        return self.last_result

    def core_polygon_projection(self, result=None):
        """Build once per routing result, then reuse across render frames."""
        result = self.last_result if result is None else result
        if result is not self._core_polygon_result:
            self._core_polygon_projection = build_core_duct_polygon_projection(
                result.supply_segments,
            )
            self._core_polygon_result = result
        return self._core_polygon_projection
