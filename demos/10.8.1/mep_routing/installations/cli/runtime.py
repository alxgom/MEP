"""Cli-owned interactive supply-tree runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from mep_routing.app import InstallationGraphProfile, create_active_dwelling
from mep_routing.geometry import largest_polygon
from mep_routing.graphs import GraphMode
from mep_routing.routing import segments_from_edges

from .geometry import cli_machine_pins, cli_supply_access_specs
from .graph_inputs import build_cli_simple_graph_inputs
from .terminals import prepare_cli_supply_terminals


@dataclass(frozen=True)
class CliRoutingResult:
    routes: tuple[tuple[str, list], ...]
    status: str
    elapsed_ms: float
    total_nodes: int
    terminals: dict[str, tuple[int, int]]
    machine_pins: dict[str, tuple[int, int]]
    workspace: object | None = None


class CliInteractiveRuntime:
    """Compose only Cli capabilities; no Sal placement or solver contracts."""

    def __init__(self, definition, graph_config, terminal_config):
        self.definition = definition
        self.graph_config = graph_config
        self.terminal_config = terminal_config
        self.last_result = CliRoutingResult((), "Cli has not run", 0.0, 0, {}, {})

    def solve(self, prepared_dwelling, machine, center, angle_degrees=0):
        started = perf_counter()
        routing_region = prepared_dwelling.routing_region_base
        terminal_region = largest_polygon(routing_region.buffer(-self.graph_config.epsilon_mm, join_style=2))
        terminals = prepare_cli_supply_terminals(
            prepared_dwelling.rooms, routing_region if terminal_region is None else terminal_region,
        )
        pins = cli_machine_pins(machine, center, angle_degrees)
        access_specs = cli_supply_access_specs(machine, pins, angle_degrees)
        if not terminals:
            return self._finish((), "Blocked: no Cli supply rooms", started, 0, terminals, pins)

        cli_prepared = replace(prepared_dwelling, terminals=dict(terminals), wet_room_names=[])
        profile = InstallationGraphProfile(
            port_access_specs=lambda current_pins, angle: cli_supply_access_specs(
                machine, current_pins, angle,
            ),
            simple_input_builder=build_cli_simple_graph_inputs,
        )
        dwelling = create_active_dwelling(
            cli_prepared, self.graph_config, profile, self.terminal_config,
        )
        build = dwelling.workspace.build_selected(GraphMode.CORE_SIMPLE, pins, angle_degrees)
        if build is None or not len(build.runtime.nodes):
            return self._finish((), "Blocked: Cli graph is empty", started, 0, terminals, pins)

        runtime = build.runtime
        required_points = [access_specs[0]["access_point"], *terminals.values()]
        terminal_nodes = tuple(int(runtime.spatial_index.query(point)[1]) for point in required_points)
        tree = self.definition.supply_tree_solver(runtime.nodes, runtime.adjacency, terminal_nodes)
        if tree is None:
            return self._finish(
                (), "Blocked: no Cli supply tree", started, len(runtime.nodes), terminals, pins,
                dwelling.workspace,
            )
        segments = list(segments_from_edges(runtime.nodes, tree.tree_edges))
        pin, access = pins["air_out"], access_specs[0]["access_point"]
        if pin != access:
            segments.insert(0, ((float(pin[0]), float(pin[1])), (float(access[0]), float(access[1]))))
        status = f"Cli supply tree solved: {len(terminals)} rooms, {len(tree.tree_edges)} edges"
        return self._finish(
            (("Supply Air Tree", segments),), status, started, len(runtime.nodes), terminals, pins,
            dwelling.workspace,
        )

    def _finish(self, routes, status, started, total_nodes, terminals, pins, workspace=None):
        self.last_result = CliRoutingResult(
            routes, status, (perf_counter() - started) * 1000.0, total_nodes,
            dict(terminals), dict(pins), workspace,
        )
        return self.last_result
