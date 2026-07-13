"""Central-plan Pygame rendering for the interactive routing workbench.

The controller prepares every domain-specific render fact (selection, graph,
machine pins, and placement guides) before calling this module.  This module
only draws screen-space primitives in a stable visual order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


Color = tuple[int, int, int]
ScreenPoint = tuple[int, int]
ScreenSegment = tuple[ScreenPoint, ScreenPoint]
WorldPoint = tuple[float, float]


@dataclass(frozen=True)
class CanvasFonts:
    small: Any


@dataclass(frozen=True)
class CanvasPolygon:
    points: Sequence[ScreenPoint]
    color: Color


@dataclass(frozen=True)
class ShaftPolygon:
    points: Sequence[ScreenPoint]
    color: Color
    source_geometry: Any
    show_hatch: bool


@dataclass(frozen=True)
class TerminalMarker:
    point: ScreenPoint
    core_color: Color
    ring_color: Color
    text_color: Color
    label: str


@dataclass(frozen=True)
class RouteStroke:
    segments: Sequence[ScreenSegment]
    width: int
    color: Color
    selected: bool


@dataclass(frozen=True)
class MachinePinMarker:
    point: ScreenPoint
    color: Color
    ring_color: Color
    radius: int


@dataclass(frozen=True)
class MachineRender:
    outline: Sequence[ScreenPoint]
    fill_color: Color
    pins: Sequence[MachinePinMarker]


@dataclass(frozen=True)
class GuideLine:
    start: ScreenPoint
    end: ScreenPoint
    color: Color
    width: int


@dataclass(frozen=True)
class CanvasScene:
    background_color: Color
    rooms: Sequence[CanvasPolygon]
    room_wall_color: Color
    room_wall_width: int
    doors: Sequence[ScreenSegment]
    door_color: Color
    columns: Sequence[CanvasPolygon]
    shafts: Sequence[ShaftPolygon]
    grid_edges: Sequence[ScreenSegment]
    grid_nodes: Sequence[ScreenPoint]
    grid_edge_color: Color
    grid_node_color: Color
    terminals: Sequence[TerminalMarker]
    shaft_marker: ScreenPoint | None
    routes: Sequence[RouteStroke]
    selection_halo_color: Color
    selected_route_name: str | None
    terminal_area_start: WorldPoint | None
    terminal_area_end: WorldPoint | None
    machine: MachineRender
    guide_lines: Sequence[GuideLine]


@dataclass(frozen=True)
class CanvasRenderHooks:
    """Adapters for existing stateful overlays and visual helpers."""

    draw_covers: Callable[[], None]
    draw_distance_heatmap: Callable[[], None]
    draw_distance_colorbar: Callable[[], None]
    draw_edge_weight_heatmap: Callable[[], None]
    draw_edge_weight_colorbar: Callable[[], None]
    draw_terminal_validity_overlay: Callable[[], None]
    draw_wet_room_accents: Callable[[], None]
    draw_polygon_hatch: Callable[[Any], None]
    draw_outlined_text: Callable[[Any, Any, str, ScreenPoint, Color], None]
    draw_preferred_terminal_areas: Callable[[str | None], None]
    draw_routed_terminal_endpoints: Callable[[str | None], None]
    draw_preferred_terminal_markers: Callable[[str | None], None]
    draw_terminal_area_drag: Callable[[WorldPoint | None, WorldPoint | None], None]


def draw_canvas_scene(
    screen: Any,
    *,
    scene: CanvasScene,
    fonts: CanvasFonts,
    hooks: CanvasRenderHooks,
) -> None:
    """Draw the central plan in the legacy layer order."""
    import pygame

    screen.fill(scene.background_color)

    for room in scene.rooms:
        pygame.draw.polygon(screen, room.color, room.points)

    hooks.draw_covers()
    hooks.draw_distance_heatmap()
    hooks.draw_distance_colorbar()
    hooks.draw_edge_weight_heatmap()
    hooks.draw_edge_weight_colorbar()
    hooks.draw_terminal_validity_overlay()
    hooks.draw_wet_room_accents()

    for room in scene.rooms:
        pygame.draw.polygon(screen, scene.room_wall_color, room.points, scene.room_wall_width)

    for start, end in scene.doors:
        pygame.draw.line(screen, scene.door_color, start, end, 4)

    for column in scene.columns:
        pygame.draw.polygon(screen, column.color, column.points)

    for shaft in scene.shafts:
        pygame.draw.polygon(screen, shaft.color, shaft.points)
        if shaft.show_hatch:
            hooks.draw_polygon_hatch(shaft.source_geometry)
            pygame.draw.polygon(screen, scene.room_wall_color, shaft.points, 1)

    for start, end in scene.grid_edges:
        pygame.draw.line(screen, scene.grid_edge_color, start, end, 1)
    for point in scene.grid_nodes:
        pygame.draw.circle(screen, scene.grid_node_color, point, 2)

    for terminal in scene.terminals:
        pygame.draw.circle(screen, terminal.ring_color, terminal.point, 7)
        pygame.draw.circle(screen, terminal.core_color, terminal.point, 5)
        label_surface = fonts.small.render(terminal.label, True, terminal.text_color)
        hooks.draw_outlined_text(
            screen,
            fonts.small,
            terminal.label,
            (terminal.point[0] - label_surface.get_width() // 2, terminal.point[1] + 10),
            terminal.text_color,
        )

    if scene.shaft_marker is not None:
        pygame.draw.circle(screen, (255, 255, 255), scene.shaft_marker, 8)
        pygame.draw.circle(screen, (231, 76, 60), scene.shaft_marker, 6)

    for route in scene.routes:
        if not route.selected:
            continue
        for start, end in route.segments:
            pygame.draw.line(screen, scene.selection_halo_color, start, end, route.width + 6)
    for route in scene.routes:
        for start, end in route.segments:
            pygame.draw.line(screen, route.color, start, end, route.width)

    hooks.draw_preferred_terminal_areas(scene.selected_route_name)
    hooks.draw_routed_terminal_endpoints(scene.selected_route_name)
    hooks.draw_preferred_terminal_markers(scene.selected_route_name)
    hooks.draw_terminal_area_drag(scene.terminal_area_start, scene.terminal_area_end)

    pygame.draw.polygon(screen, scene.machine.fill_color, scene.machine.outline)
    pygame.draw.polygon(screen, (255, 255, 255), scene.machine.outline, 2)
    for pin in scene.machine.pins:
        pygame.draw.circle(screen, pin.color, pin.point, pin.radius)
        pygame.draw.circle(screen, pin.ring_color, pin.point, pin.radius, 1)

    for guide in scene.guide_lines:
        pygame.draw.line(screen, guide.color, guide.start, guide.end, guide.width)
