"""Mutable graph-session state for the interactive routing application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .weight_runtime import EdgeWeightOverlay, StaticClearanceCache


@dataclass
class RoutingWorkspace:
    """Own selected, dynamic, and placement graphs for one dwelling session."""

    lifecycle: Any = None
    active_graph: Any = None
    env: Any = None
    base_regular: Any = None
    static_clearance_cache: StaticClearanceCache = field(default_factory=StaticClearanceCache)
    overlay: EdgeWeightOverlay = field(default_factory=EdgeWeightOverlay)

    @property
    def grid_available(self) -> bool:
        return self.active_graph is not None

    @property
    def nodes(self):
        return None if self.active_graph is None else self.active_graph.nodes

    @property
    def edge_list(self):
        return () if self.active_graph is None else self.active_graph.edge_list

    @property
    def edge_coords(self):
        return None if self.active_graph is None else self.active_graph.edge_coords

    @property
    def spatial_index(self):
        return None if self.active_graph is None else self.active_graph.spatial_index

    @property
    def base_env(self):
        return None if self.base_regular is None else self.base_regular.env

    @property
    def base_spatial_index(self):
        return None if self.base_regular is None else self.base_regular.spatial_index

    def replace_dwelling(self, lifecycle: Any) -> None:
        self.lifecycle = lifecycle
        self.active_graph = None
        self.env = None
        self.base_regular = None
        self.static_clearance_cache = StaticClearanceCache()
        self.overlay = EdgeWeightOverlay()

    def reset_routing_fields(self) -> None:
        self.static_clearance_cache = StaticClearanceCache()
        self.overlay.reset()

    def commit(self, runtime: Any, terminal_runtime: Any = None) -> None:
        self.active_graph = runtime
        self.env = runtime.env
        self.static_clearance_cache = StaticClearanceCache()
        self._sync_terminal_runtime(terminal_runtime)

    def build_base_regular(self):
        self.base_regular = None if self.lifecycle is None else self.lifecycle.build_base_regular()
        return self.base_regular

    def build_selected(self, graph_type, pins, angle, *, shift_hannan_walls=True, terminal_runtime=None):
        if self.lifecycle is None:
            return None
        result = self.lifecycle.build_selected(
            graph_type, pins, angle, shift_hannan_walls=shift_hannan_walls,
        )
        if result is not None:
            self.commit(result.runtime, terminal_runtime)
        return result

    def apply_machine_obstacle(self, polygon, pins, angle, *, clearance_mm, terminal_runtime=None):
        if self.lifecycle is None or self.active_graph is None:
            self.env = None
            self._sync_terminal_runtime(terminal_runtime)
            return None
        result = self.lifecycle.apply_dynamic_obstacle(
            self.active_graph, polygon, pins, angle, clearance_mm=clearance_mm,
        )
        self.env = result.env
        self._sync_terminal_runtime(terminal_runtime)
        return result

    def _sync_terminal_runtime(self, terminal_runtime: Any) -> None:
        if terminal_runtime is None:
            return
        if self.env is None:
            terminal_runtime.set_graph(None, None, None)
            return
        terminal_runtime.set_graph(self.env.nodes, self.env.adj, self.spatial_index)
