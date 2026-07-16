"""Per-dwelling shaft-entry selection cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .shaft_entries import select_shaft_entry_nodes, shaft_entry_geometry, shaft_entry_segments


@dataclass
class ShaftEntryRuntime:
    shaft: Any
    core_entry_specs: tuple = ()
    geometry_by_node: dict[int, dict] = field(default_factory=dict)

    def geometry(self, node_index, env):
        if node_index in self.geometry_by_node:
            return self.geometry_by_node[node_index]
        if self.shaft is None or env is None or node_index is None:
            return None
        return shaft_entry_geometry(self.shaft, env.nodes[int(node_index)])

    def select(self, env, *, spatial_index=None, search_radius_mm, grid_spacing_mm, max_candidates):
        if self.shaft is None or env is None or not len(env.nodes):
            self.geometry_by_node = {}
            return [], None
        candidates, chosen, self.geometry_by_node = select_shaft_entry_nodes(
            env.nodes, self.shaft,
            search_radius_mm=search_radius_mm,
            grid_spacing_mm=grid_spacing_mm,
            max_candidates=max_candidates,
            core_entry_specs=self.core_entry_specs,
            spatial_index=spatial_index,
        )
        return candidates, chosen

    def append_segments(self, segments, node_index, env):
        segments.extend(shaft_entry_segments(self.geometry(node_index, env)))
