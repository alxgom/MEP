"""Cli-owned interactive supply and return-air runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from mep_routing.app import InstallationGraphProfile, create_active_dwelling
from mep_routing.graphs import GraphMode

from .geometry import cli_air_access_spec, cli_machine_pins
from .graph_inputs import build_cli_simple_graph_inputs
from .grilles import CliGrillePlan, prepare_cli_grilles
from .materializer import CliDuctSegment, materialize_cli_tree
from .placement import CliPlacementResult, CliPlacementRuntime


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

    def __init__(self, definition, graph_config, terminal_config, placement=None):
        self.definition = definition
        self.graph_config = graph_config
        self.terminal_config = terminal_config
        self.placement = placement or CliPlacementRuntime()
        self.last_placement = None
        self.last_result = CliRoutingResult((), "Cli has not run", 0.0, 0, {}, {})

    def place_machine(self, prepared_dwelling, machine, *, context_obstacles=()):
        grilles = prepare_cli_grilles(
            prepared_dwelling.rooms, prepared_dwelling.routing_region_base, machine=machine,
        )
        self.last_placement = self.placement.place(
            prepared_dwelling, machine, grilles, context_obstacles=context_obstacles,
        )
        return self.last_placement

    def solve(self, prepared_dwelling, machine, center, angle_degrees=0, *, context_obstacles=()):
        started = perf_counter()
        pins = cli_machine_pins(machine, center, angle_degrees)
        grilles = prepare_cli_grilles(
            prepared_dwelling.rooms, prepared_dwelling.routing_region_base,
            machine=machine, machine_pins=pins, angle_degrees=angle_degrees,
        )
        if not grilles.supply:
            return self._finish(
                (), "Blocked: no Cli supply grilles", started, 0, grilles, pins,
                context_obstacles=context_obstacles,
            )

        supply = self._solve_tree(
            "supply", prepared_dwelling, machine, pins, angle_degrees,
            grilles, context_obstacles,
        )
        if supply is None:
            if context_obstacles:
                relaxed = self.solve(prepared_dwelling, machine, center, angle_degrees)
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
        returned = self._solve_tree(
            "return", prepared_dwelling, machine, pins, angle_degrees,
            grilles, context_obstacles,
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
        else:
            status += "; return tree blocked"
        return self._finish(
            routes, status, started, total_nodes, grilles, pins,
            workspace=supply.workspace, supply_segments=supply.segments,
            return_segments=return_segments, context_obstacles=context_obstacles,
        )

    def _solve_tree(self, role, prepared, machine, pins, angle, grilles, context_obstacles):
        connections = grilles.supply if role == "supply" else grilles.returns
        if not connections:
            return None
        connector_key = "air_out" if role == "supply" else "air_in"
        access = cli_air_access_spec(machine, pins, angle, connector_key)
        terminals = {item.route_name: item.routing_point for item in connections}
        role_prepared = replace(prepared, terminals=terminals, wet_room_names=[])

        def simple_inputs(**kwargs):
            return build_cli_simple_graph_inputs(
                **kwargs, context_obstacles=context_obstacles,
            )

        profile = InstallationGraphProfile(
            port_access_specs=lambda current_pins, current_angle: (
                cli_air_access_spec(machine, current_pins, current_angle, connector_key),
            ),
            simple_input_builder=simple_inputs,
        )
        dwelling = create_active_dwelling(
            role_prepared, self.graph_config, profile, self.terminal_config,
        )
        build = dwelling.workspace.build_selected(GraphMode.CORE_SIMPLE, pins, angle)
        if build is None or not len(build.runtime.nodes):
            return None
        runtime = build.runtime
        points = [access["access_point"], *terminals.values()]
        indices = tuple(dict.fromkeys(int(runtime.spatial_index.query(point)[1]) for point in points))
        tree = self.definition.supply_tree_solver(runtime.nodes, runtime.adjacency, indices)
        if tree is None:
            return None
        segments = materialize_cli_tree(
            role=role, machine=machine, pins=pins, access_spec=access,
            graph_nodes=runtime.nodes, tree=tree, grille_connections=connections,
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
        return self.last_result
