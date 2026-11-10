"""Narrow application lifecycle for installation-owned routing runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from mep_routing.geometry import iter_polygons
from mep_routing.installations.cli import cli_machine_pins
from mep_routing.installations.cli.geometry import (
    cli_clear_supply_turns, cli_machine_footprint, cli_supply_access_specs, cli_supply_exit_footprints,
)
from mep_routing.installations.cli.placement_mask import CliPlacementMasks


@dataclass(frozen=True)
class InstallationCapabilities:
    """Interaction features exposed by an installation, independent of its solver."""

    machine_selection: bool = False
    automatic_placement: bool = False
    terminal_tools: bool = False
    machine_edit: bool = True
    solver_controls: bool = False
    route_validation: bool = False
    placement_overlay: bool = False
    routing_weights: bool = False


@dataclass(frozen=True)
class InstallationSolveContext:
    dwelling: object
    installation_state: object
    context_obstacles: tuple


@runtime_checkable
class InstallationRuntime(Protocol):
    """Shared lifecycle only; graph, placement, and solver APIs remain concrete."""

    key: str
    capabilities: InstallationCapabilities

    def solve(self, context: InstallationSolveContext) -> object: ...

    def machine_pins(self, state: object) -> Mapping[str, tuple[float, float]]: ...


class InstallationRegistry:
    def __init__(self, runtimes):
        self._runtimes = {runtime.key: runtime for runtime in runtimes}

    def get(self, key: str) -> InstallationRuntime | None:
        return self._runtimes.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._runtimes


class SalInstallationRuntime:
    key = "sal"
    capabilities = InstallationCapabilities(
        automatic_placement=True, terminal_tools=True,
        solver_controls=True, route_validation=True, placement_overlay=True,
        routing_weights=True,
    )

    def __init__(self, controller):
        self.controller = controller

    def solve(self, context):
        return self.controller.interactive_solver(context.context_obstacles).solve()

    def machine_pins(self, state):
        return self.controller.machine_pins(*state.pose.as_tuple())


class CliInstallationRuntime:
    key = "cli"
    capabilities = InstallationCapabilities(
        machine_selection=True, automatic_placement=True,
        routing_weights=True,
    )

    def __init__(self, app):
        self.app = app
        self._placement_masks = CliPlacementMasks()
        self._cell_dwelling = None
        self._placement_cell_grids = {}

    def placement_mask(self, dwelling, angle_degrees):
        machine = self._machine()
        return self._placement_masks.get(dwelling, machine, angle_degrees, self._exit_dimensions(machine))

    def _exit_dimensions(self, machine):
        return self.app.cli_runtime.supply_exit_dimensions(
            machine, self.app.workbench_state.cli.minimum_segment_length_factor,
        )

    def placement_cell_grid(self, dwelling, *, room_aware):
        """Cache both placement overlays independently of the machine pose."""
        if dwelling is not self._cell_dwelling:
            self._placement_cell_grids.clear()
            self._cell_dwelling = dwelling
        if room_aware not in self._placement_cell_grids:
            rooms = tuple(room.polygon for room in dwelling.prepared.rooms) if room_aware else ()
            self._placement_cell_grids[room_aware] = dwelling.workspace.lifecycle.build_static_cell_grid(
                room_polys=rooms,
            )
        return self._placement_cell_grids[room_aware]

    def placement_outlines(self, dwelling, pins, angle_degrees):
        machine = self._machine()
        length, width = self._exit_dimensions(machine)
        access, = cli_supply_access_specs(machine, pins, angle_degrees, length)
        extension, turns = cli_supply_exit_footprints(machine, angle_degrees, access, width)
        usable_turns = cli_clear_supply_turns(dwelling, extension, turns)
        footprint = cli_machine_footprint(pins).union(extension)
        for turn in usable_turns.values():
            footprint = footprint.union(turn)
        return tuple(tuple(part.exterior.coords) for part in iter_polygons(footprint))

    def _machine(self):
        key = self.app.installation_ui_session.selected_keys[self.key]
        return self.app.CLI_INSTALLATION.machine_catalog.value(key)

    def solve(self, context):
        machine = self._machine()
        pose = context.installation_state.pose
        return self.app.cli_runtime.solve(
            context.dwelling.prepared,
            machine,
            pose.center,
            pose.angle_degrees,
            context_obstacles=context.context_obstacles,
            return_enabled=self.app.workbench_state.cli.return_ventilation_enabled,
            graph_mode=self.app.CLI_INSTALLATION.graph_modes[
                self.app.workbench_state.cli.graph_type_idx
            ],
            solver_backend_index=self.app.workbench_state.cli.router_backend_idx,
            start_mode_index=self.app.workbench_state.cli.room_start_mode_idx,
            minimum_segment_length_factor=self.app.workbench_state.cli.minimum_segment_length_factor,
            bend_penalty_mm=self.app.workbench_state.cli.bend_penalty_mm,
            crossing_penalty_multiplier=(
                self.app.workbench_state.cli.crossing_penalty_multiplier
            ),
        )

    def machine_pins(self, state):
        return cli_machine_pins(self._machine(), state.pose.center, state.pose.angle_degrees)


class CocInstallationRuntime:
    key = "coc"
    capabilities = InstallationCapabilities(
        machine_edit=False, solver_controls=True,
    )

    def __init__(self, app):
        self.app = app

    def solve(self, context):
        state = self.app.workbench_state.coc
        return self.app.coc_runtime.solve(
            context.dwelling.prepared,
            context_obstacles=context.context_obstacles,
            graph_mode=self.app.COC_INSTALLATION.graph_modes[state.graph_type_idx],
        )

    def machine_pins(self, state):
        return self.app.coc_runtime.last_result.machine_pins
