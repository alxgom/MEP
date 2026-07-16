"""Active dwelling geometry and mutable routing runtimes."""

from __future__ import annotations

from dataclasses import dataclass

from mep_routing.routing import RoutingWorkspace, TerminalRuntime


@dataclass
class ActiveDwellingSession:
    """One prepared dwelling paired with its graph and terminal runtimes."""

    prepared: object
    workspace: RoutingWorkspace
    terminals: TerminalRuntime

    @classmethod
    def create(
        cls,
        prepared,
        lifecycle,
        *,
        regulation_clearance_mm,
        terminal_buffer_mm,
        remap_tolerance_mm,
    ):
        terminal_runtime = TerminalRuntime(
            terminals=prepared.terminals,
            room_polygons={room.name: room.polygon for room in prepared.rooms},
            routing_region=prepared.routing_region_base,
            covers=prepared.covers,
            walls=prepared.walls,
            wall_polygons=prepared.wall_polygons,
            regulation_clearance_mm=regulation_clearance_mm,
            terminal_buffer_mm=terminal_buffer_mm,
            remap_tolerance_mm=remap_tolerance_mm,
        )
        workspace = RoutingWorkspace()
        workspace.replace_dwelling(lifecycle)
        return cls(prepared, workspace, terminal_runtime)

    @property
    def rooms(self):
        return self.prepared.rooms

    @property
    def columns(self):
        return self.prepared.columns

    @property
    def shafts(self):
        return self.prepared.shafts

    @property
    def covers(self):
        return self.prepared.covers

    @property
    def doors(self):
        return self.prepared.doors

    @property
    def walls(self):
        return self.prepared.walls

    @property
    def wall_polygons(self):
        return self.prepared.wall_polygons

    @property
    def routing_region(self):
        return self.prepared.routing_region_base

    @property
    def shaft_extraction(self):
        return self.prepared.shaft_extraction

    @property
    def terminal_points(self):
        return self.prepared.terminals

    @property
    def wet_room_names(self):
        return self.prepared.wet_room_names

    @property
    def shaft_core_entry_specs(self):
        return self.prepared.shaft_core_entry_specs
