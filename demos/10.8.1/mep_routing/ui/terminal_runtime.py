"""Bound terminal interaction and rendering state for the Pygame workbench."""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Point

from mep_routing.routing import terminal_validity_entries
from .drawing import draw_dashed_polyline
from .terminal_selection import (
    draw_preferred_terminal_areas,
    draw_preferred_terminal_markers,
    draw_routed_terminal_endpoint_markers,
)
from .terminal_validity import (
    draw_terminal_validity_overlay,
    draw_terminal_validity_square,
    draw_terminal_validity_tooltip,
)


@dataclass(frozen=True)
class TerminalUiStyle:
    route_colors: dict
    text_color: tuple[int, int, int]
    allowed_color: tuple[int, int, int]
    blocked_color: tuple[int, int, int]
    blocked_hatch_color: tuple[int, int, int]
    marker_size_mm: float
    remap_tolerance_mm: float
    regulation_clearance_mm: float
    room_buffer_mm: float


class TerminalUiRuntime:
    """Own terminal UI cache and adapt a dwelling session to Pygame drawing."""

    def __init__(self, canvas, style: TerminalUiStyle):
        self.canvas = canvas
        self.style = style
        self.dwelling = None
        self.terminals = None
        self.workspace = None
        self._validity_key = None
        self._validity_value = ([], {})

    def bind(self, dwelling) -> None:
        self.dwelling = dwelling
        self.terminals = None if dwelling is None else dwelling.terminals
        self.workspace = None if dwelling is None else dwelling.workspace
        self.invalidate()

    def invalidate(self) -> None:
        self._validity_key = None
        self._validity_value = ([], {})

    @property
    def ready(self) -> bool:
        return self.terminals is not None and self.workspace is not None

    def marker_side(self) -> int:
        return max(4, int(round(self.style.marker_size_mm * self.canvas.viewport.scale)))

    def candidate_nodes(self, route_name):
        return [] if not self.ready else self.terminals.candidate_nodes(route_name)

    def route_start_nodes(self, route_name, *, use_nearest_terminal=False):
        if not self.ready:
            return []
        return self.terminals.route_start_nodes(route_name, use_nearest_terminal=use_nearest_terminal)

    def find_candidate(self, world_point):
        return None if not self.ready else self.terminals.find_candidate_at(world_point)

    def apply_point_preference(self, world_point, *, remove=False):
        if not self.ready:
            return False, None
        return self.terminals.apply_point_preference(world_point, remove=remove)

    def apply_area_preference(self, start_world, end_world, *, remove=False):
        if not self.ready:
            return False, None
        return self.terminals.apply_area_preference(start_world, end_world, remove=remove)

    def route_room_polygon(self, route_name):
        return None if not self.ready else self.terminals.room_polygon(route_name)

    def find_room_route(self, world_point, route_names, shaft=None):
        point = Point(float(world_point[0]), float(world_point[1]))
        tolerance = max(40.0, 8.0 / self.canvas.viewport.scale)
        if "Shaft" in route_names and shaft is not None:
            if shaft.contains(point) or shaft.distance(point) < tolerance:
                return "Shaft"
        candidates = []
        for route_name in route_names:
            polygon = self.route_room_polygon(route_name)
            if polygon is not None and (polygon.contains(point) or polygon.distance(point) < 1e-7):
                candidates.append((polygon.area, route_name))
        return min(candidates)[1] if candidates else None

    def validity_entries(self):
        env = None if self.workspace is None else self.workspace.env
        if env is None or self.terminals is None:
            return [], {}
        key = (id(env.nodes), len(env.nodes), tuple(sorted(self.dwelling.terminal_points)), id(self.dwelling.routing_region))
        if key != self._validity_key:
            self._validity_value = terminal_validity_entries(
                env.nodes, env.adj, self.dwelling.terminal_points.keys(), self.dwelling.routing_region,
                self.terminals.room_polygon, self.terminals.valid_region,
                self.terminals.boundary_segments, self.style.regulation_clearance_mm,
                self.style.room_buffer_mm,
            )
            self._validity_key = key
        return self._validity_value

    def draw_preferred_areas(self, screen, selected_route_name=None):
        env = None if self.workspace is None else self.workspace.env
        if env is None or self.terminals is None:
            return
        return draw_preferred_terminal_areas(
            screen, self.terminals.preferred_areas, selected_route_name, self.style.route_colors,
            self.terminals.candidate_nodes, env.nodes, self.canvas.viewport.to_screen,
            max(3, self.marker_side() // 2),
        )

    def draw_preferred_markers(self, screen, selected_route_name=None, routes=None):
        env = None if self.workspace is None else self.workspace.env
        if env is None or self.terminals is None:
            return
        return draw_preferred_terminal_markers(
            screen, self.dwelling.terminal_points.keys(), self.terminals.preferred_points_by_room,
            selected_route_name, routes, self.style.route_colors, self.style.text_color,
            self.terminals.candidate_nodes, env.nodes, self.canvas.viewport.to_screen,
            self.marker_side(), self.style.remap_tolerance_mm,
        )

    def draw_routed_endpoints(self, screen, routes, selected_route_name=None):
        route_names = () if self.dwelling is None else self.dwelling.terminal_points.keys()
        return draw_routed_terminal_endpoint_markers(
            screen, routes, route_names, selected_route_name, self.style.route_colors,
            self.style.text_color, self.canvas.viewport.to_screen, self.marker_side(),
        )

    def draw_validity_square(self, screen, center, side, allowed):
        return draw_terminal_validity_square(
            screen, center, side, allowed, self.style.allowed_color, self.style.blocked_color,
            self.style.blocked_hatch_color, draw_dashed_polyline,
        )

    def draw_validity_overlay(self, screen, enabled=True):
        if not enabled:
            return
        entries, _ = self.validity_entries()
        marker_side = max(4, min(13, int(round(70 * self.canvas.viewport.scale))))
        return draw_terminal_validity_overlay(
            screen, entries, self.canvas.viewport.to_screen, self.canvas.viewport.canvas_rect,
            marker_side, self.style.allowed_color, self.style.blocked_color,
            self.style.blocked_hatch_color, draw_dashed_polyline,
        )

    def draw_validity_tooltip(self, screen, font, mouse_position, enabled=True):
        env = None if self.workspace is None else self.workspace.env
        spatial_index = None if self.workspace is None else self.workspace.spatial_index
        if not enabled or env is None or spatial_index is None:
            return
        _, reasons = self.validity_entries()
        return draw_terminal_validity_tooltip(
            screen, font, mouse_position, self.canvas.viewport.canvas_rect,
            self.canvas.viewport.to_world, lambda point: int(spatial_index.query(point)[1]),
            env.nodes, reasons, self.canvas.viewport.to_screen, self.canvas.viewport.window_size,
            self.style.text_color,
        )
