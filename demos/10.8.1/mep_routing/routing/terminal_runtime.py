"""Live terminal-start and preference coordination without UI dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Sequence

from shapely.geometry import Point

from mep_routing.ui.terminal_selection import (
    apply_preferred_terminal_area,
    apply_preferred_terminal_point,
    find_room_candidate_node,
    map_preferred_points_to_nodes,
)

from .terminal_regions import (
    room_cover_geometry,
    terminal_boundary_segments,
    terminal_valid_region,
)
from .terminal_validity import terminal_candidate_node_indices


@dataclass
class TerminalRuntime:
    """Own terminal candidate caches and preferred-start selections for one dwelling.

    The caller updates graph or geometry inputs explicitly through ``set_graph``
    and ``set_geometry``. This keeps cache invalidation local to terminal policy
    while leaving application state and rendering outside this module.
    """

    terminals: Mapping[str, tuple[float, float]]
    room_polygons: Mapping[str, Any]
    routing_region: Any
    covers: Sequence[Any]
    walls: Sequence[Any]
    wall_polygons: Sequence[Any]
    regulation_clearance_mm: float
    terminal_buffer_mm: float
    remap_tolerance_mm: float
    nodes: Any = None
    adjacency: Mapping[int, Any] | None = None
    nearest_index: Any = None
    preferred_points_by_room: MutableMapping[str, list[tuple[float, float]]] = field(default_factory=dict)
    preferred_areas: list[dict[str, Any]] = field(default_factory=list)
    _candidate_cache: dict[str, tuple[int, ...]] = field(default_factory=dict, init=False, repr=False)
    _cover_cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _boundary_cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def set_graph(self, nodes, adjacency, nearest_index) -> None:
        """Replace the active graph inputs and invalidate graph-derived candidates."""
        self.nodes = nodes
        self.adjacency = adjacency
        self.nearest_index = nearest_index
        self.invalidate_candidates()

    def set_geometry(
        self,
        *,
        room_polygons: Mapping[str, Any],
        routing_region: Any,
        covers: Sequence[Any],
        walls: Sequence[Any],
        wall_polygons: Sequence[Any],
    ) -> None:
        """Replace dwelling geometry and invalidate all terminal geometry caches."""
        self.room_polygons = room_polygons
        self.routing_region = routing_region
        self.covers = covers
        self.walls = walls
        self.wall_polygons = wall_polygons
        self.invalidate_geometry()

    def invalidate_candidates(self) -> None:
        """Clear candidate nodes after an environment or graph change."""
        self._candidate_cache.clear()

    def invalidate_geometry(self) -> None:
        """Clear all cached terminal geometry after dwelling geometry changes."""
        self._cover_cache.clear()
        self._boundary_cache.clear()
        self.invalidate_candidates()

    def clear_preferences(self) -> None:
        """Remove saved point and area preferences for a new dwelling session."""
        self.preferred_points_by_room.clear()
        self.preferred_areas.clear()

    def restore_preferences(self, points_by_room, areas) -> None:
        """Replace saved preferences from a restored solution snapshot."""
        self.preferred_points_by_room.clear()
        self.preferred_points_by_room.update({
            room_name: [tuple(point) for point in points]
            for room_name, points in points_by_room.items()
        })
        self.preferred_areas[:] = [dict(area) for area in areas]

    def room_polygon(self, room_name: str):
        return self.room_polygons.get(room_name)

    def room_cover(self, room_name: str):
        if room_name not in self._cover_cache:
            self._cover_cache[room_name] = room_cover_geometry(self.room_polygon(room_name), self.covers)
        return self._cover_cache[room_name]

    def valid_region(self, room_name: str):
        return terminal_valid_region(
            self.room_polygon(room_name), self.routing_region, self.room_cover(room_name),
        )

    def boundary_segments(self, room_name: str):
        if room_name not in self._boundary_cache:
            self._boundary_cache[room_name] = terminal_boundary_segments(
                self.room_polygon(room_name),
                tuple(self.room_polygons.values()),
                self.walls,
                self.wall_polygons,
                self.room_cover(room_name),
            )
        return self._boundary_cache[room_name]

    def candidate_nodes(self, route_name: str) -> list[int]:
        """Return cached, valid terminal start nodes nearest the terminal first."""
        if self.nodes is None or self.adjacency is None or self.nearest_index is None:
            return []
        cached = self._candidate_cache.get(route_name)
        if cached is not None:
            return list(cached)

        terminal_point = self.terminals.get(route_name)
        if terminal_point is None:
            return []

        room_polygon = self.room_polygon(route_name)
        if room_polygon is None or self.routing_region is None:
            _distance, node_index = self.nearest_index.query(terminal_point)
            result = (int(node_index),)
        else:
            valid_region = self.valid_region(route_name)
            if valid_region is None or valid_region.is_empty:
                result = ()
            else:
                result = tuple(terminal_candidate_node_indices(
                    self.nodes,
                    self.adjacency,
                    valid_region,
                    terminal_point,
                    self.boundary_segments(route_name),
                    max(self.regulation_clearance_mm, self.terminal_buffer_mm),
                ))
        self._candidate_cache[route_name] = result
        return list(result)

    def preferred_nodes(self, route_name: str) -> list[int]:
        """Map saved preference points to current candidate nodes."""
        if self.nodes is None:
            return []
        nodes, _preference_indices = map_preferred_points_to_nodes(
            self.preferred_points_by_room.get(route_name, []),
            self.candidate_nodes(route_name),
            self.nodes,
            self.remap_tolerance_mm,
        )
        return nodes

    def route_start_nodes(self, route_name: str, *, use_nearest_terminal: bool = False) -> list[int]:
        """Choose nearest, all-valid, or explicitly preferred start nodes."""
        if self.nodes is None or self.nearest_index is None or route_name not in self.terminals:
            return []
        if use_nearest_terminal:
            _distance, node_index = self.nearest_index.query(self.terminals[route_name])
            return [int(node_index)]
        candidates = self.candidate_nodes(route_name)
        if self.preferred_points_by_room.get(route_name):
            return self.preferred_nodes(route_name)
        return candidates

    def room_contains_point(self, room_name: str, world_point) -> bool:
        """Return whether a world-space point falls inside a terminal room."""
        room_polygon = self.room_polygon(room_name)
        if room_polygon is None:
            return False
        point = Point(float(world_point[0]), float(world_point[1]))
        return bool(room_polygon.contains(point) or room_polygon.distance(point) < 1e-7)

    def find_candidate_at(self, world_point):
        """Return the nearest valid terminal candidate in the containing room."""
        if self.nodes is None:
            return None
        return find_room_candidate_node(
            world_point,
            self.terminals.keys(),
            self.room_contains_point,
            self.candidate_nodes,
            self.nodes,
        )

    def apply_point_preference(self, world_point, *, remove: bool = False):
        """Add or remove the nearest valid terminal preference in place."""
        if self.nodes is None:
            return False, None
        return apply_preferred_terminal_point(
            self.preferred_points_by_room,
            world_point,
            remove,
            self.terminals.keys(),
            self.room_contains_point,
            self.candidate_nodes,
            self.nodes,
            self.remap_tolerance_mm,
        )

    def apply_area_preference(self, start_world, end_world, *, remove: bool = False):
        """Add or remove valid terminal preferences in a rectangular area."""
        if self.nodes is None:
            return False, None
        return apply_preferred_terminal_area(
            self.preferred_points_by_room,
            self.preferred_areas,
            start_world,
            end_world,
            remove,
            self.terminals.keys(),
            self.candidate_nodes,
            self.nodes,
            self.remap_tolerance_mm,
        )
