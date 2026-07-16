"""Pure view-model builders for one interactive-router frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .canvas import (
    CanvasPolygon,
    CanvasScene,
    GuideLine,
    MachinePinMarker,
    MachineRender,
    RouteStroke,
    ShaftPolygon,
    TerminalMarker,
)
from .sidebar import AutoPlacementCard, ExecutionStatusCard, MachineCard, SidebarView, SolverCard


Color = tuple[int, int, int]


@dataclass(frozen=True)
class CanvasFramePalette:
    background: Color
    wall: Color
    door: Color
    column: Color
    shaft: Color
    inactive_shaft: Color
    graph_edge: Color
    graph_node: Color
    room: Color
    covered_room: Color
    deselected_room: Color
    deselected_pin: Color
    deselected_route: Color
    plan_label: Color
    text: Color
    selection_halo: Color


@dataclass(frozen=True)
class CanvasFrameCallbacks:
    to_screen: Callable[[float, float], tuple[int, int]]
    representative_point: Callable[[Any], tuple[float, float]]
    route_draw_width: Callable[[str], int]
    spatial_query: Callable[[tuple[float, float]], tuple[Any, int]]


@dataclass(frozen=True)
class CanvasFrameState:
    rooms: Sequence[Any]
    doors: Sequence[Mapping[str, Any]]
    columns: Sequence[Any]
    shafts: Sequence[Any]
    shaft_extraction: Any
    graph_env: Any
    show_grid_graph: bool
    terminals: Mapping[str, tuple[float, float]]
    routes: Sequence[tuple[str, Sequence]]
    selected_route_name: str | None
    selected_room_polygon: Any
    global_pins: Mapping[str, tuple[float, float]]
    selected_pins: set[str]
    auto_placement_mode: int
    placement_fields: Mapping[str, Mapping[int, float]]
    wet_room_names: Sequence[str]
    route_colors: Mapping[str, Color]
    palette: CanvasFramePalette
    callbacks: CanvasFrameCallbacks
    room_wall_width: int
    terminal_area_start: tuple[float, float] | None
    terminal_area_end: tuple[float, float] | None


def active_selected_route(selected_route_name, routes):
    """Keep selection only while a matching rendered route exists."""
    if selected_route_name and routes and any(name == selected_route_name for name, _ in routes):
        return selected_route_name
    return None


def build_canvas_scene(state: CanvasFrameState) -> CanvasScene:
    """Derive screen-space canvas primitives without invoking Pygame."""
    palette, callbacks = state.palette, state.callbacks
    to_screen = callbacks.to_screen

    canvas_rooms = []
    for room in state.rooms:
        if not hasattr(room, "polygon") or room.polygon.is_empty:
            continue
        room_name = getattr(room, "name", None)
        is_selected = state.selected_route_name and (
            room_name == state.selected_route_name
            or (state.selected_room_polygon is not None and room.polygon.equals(state.selected_room_polygon))
        )
        color = (
            palette.deselected_room if state.selected_route_name and not is_selected
            else palette.covered_room if room.has_cover else palette.room
        )
        canvas_rooms.append(CanvasPolygon(
            [to_screen(x, y) for x, y in room.polygon.exterior.coords], color,
        ))

    canvas_doors = [
        (to_screen(*door["d1"]), to_screen(*door["d2"])) for door in state.doors
    ]
    canvas_columns = [
        CanvasPolygon([to_screen(x, y) for x, y in column.exterior.coords], palette.column)
        for column in state.columns
    ]
    canvas_shafts = []
    for shaft in state.shafts:
        active = state.shaft_extraction is not None and shaft.equals(state.shaft_extraction)
        canvas_shafts.append(ShaftPolygon(
            [to_screen(x, y) for x, y in shaft.exterior.coords],
            palette.shaft if active else palette.inactive_shaft,
            shaft,
            not active,
        ))

    grid_edges, grid_nodes = [], []
    if state.show_grid_graph and state.graph_env is not None:
        env = state.graph_env
        grid_edges = [
            (to_screen(*env.nodes[u][:2]), to_screen(*env.nodes[v][:2]))
            for u in env.adj for v, _distance, _direction in env.adj[u] if u < v
        ]
        grid_nodes = [to_screen(*point[:2]) for point in env.nodes]

    terminal_markers = []
    for route_name, point in state.terminals.items():
        core = state.route_colors.get(route_name, (255, 255, 255))
        if state.selected_route_name and route_name != state.selected_route_name:
            core, ring, text = palette.deselected_pin, (70, 74, 78), (84, 88, 94)
        else:
            ring, text = (255, 255, 255), palette.plan_label
        terminal_markers.append(TerminalMarker(
            to_screen(*point), core, ring, text,
            route_name.replace("Bathroom", "Bath").replace("Washroom", "Wash"),
        ))

    shaft_marker = None
    if state.shaft_extraction is not None:
        shaft_marker = to_screen(*callbacks.representative_point(state.shaft_extraction))

    route_strokes = []
    for route_name, segments in state.routes or ():
        color = state.route_colors.get(route_name, palette.text)
        if state.selected_route_name and state.selected_route_name != route_name:
            color = palette.deselected_route
        route_strokes.append(RouteStroke(
            [(to_screen(*p1), to_screen(*p2)) for p1, p2 in segments],
            callbacks.route_draw_width(route_name), color,
            state.selected_route_name == route_name,
        ))

    outline = [to_screen(*state.global_pins[name]) for name in ("c_tl", "c_tr", "c_br", "c_bl")]
    pin_markers = []
    for pin_name in ("tl", "tr", "bl", "br", "left_mid", "right_mid"):
        large = pin_name in ("left_mid", "right_mid")
        color, ring = ((241, 196, 15) if large else (230, 126, 34)), (255, 255, 255)
        if state.selected_route_name and pin_name not in state.selected_pins:
            color, ring = palette.deselected_pin, (80, 84, 88)
        pin_markers.append(MachinePinMarker(to_screen(*state.global_pins[pin_name]), color, ring, 5 if large else 4))

    guides = _build_guide_lines(state)
    return CanvasScene(
        palette.background, canvas_rooms, palette.wall, state.room_wall_width,
        canvas_doors, palette.door, canvas_columns, canvas_shafts,
        grid_edges, grid_nodes, palette.graph_edge, palette.graph_node,
        terminal_markers, shaft_marker, route_strokes, palette.selection_halo,
        state.selected_route_name, state.terminal_area_start, state.terminal_area_end,
        MachineRender(outline, (230, 126, 34) if state.auto_placement_mode > 0 else (127, 140, 141), pin_markers),
        guides,
    )


def _build_guide_lines(state: CanvasFrameState):
    if state.auto_placement_mode != 1 or not state.placement_fields:
        return []
    to_screen, query = state.callbacks.to_screen, state.callbacks.spatial_query
    shaft_point = state.callbacks.representative_point(state.shaft_extraction)
    _, left = query(state.global_pins["left_mid"])
    _, right = query(state.global_pins["right_mid"])
    shaft_field = state.placement_fields["Shaft"]
    left_distance, right_distance = shaft_field.get(int(left), 1e9), shaft_field.get(int(right), 1e9)
    exhaust = "left_mid" if left_distance < right_distance else "right_mid"
    kitchen = "right_mid" if left_distance < right_distance else "left_mid"
    guides = [GuideLine(to_screen(*state.global_pins[exhaust]), to_screen(*shaft_point), (46, 204, 113), 2)]
    if "Kitchen" in state.terminals:
        guides.append(GuideLine(to_screen(*state.global_pins[kitchen]), to_screen(*state.terminals["Kitchen"]), (241, 196, 15), 2))
    used = set()
    for route_name in (name for name in state.wet_room_names if name != "Kitchen"):
        best_pin, best_distance = None, 1e9
        for pin_name in ("tl", "tr", "bl", "br"):
            if pin_name in used:
                continue
            _, index = query(state.global_pins[pin_name])
            distance = state.placement_fields[route_name].get(int(index), 1e9)
            if distance < best_distance:
                best_distance, best_pin = distance, pin_name
        if best_pin:
            used.add(best_pin)
            guides.append(GuideLine(
                to_screen(*state.global_pins[best_pin]), to_screen(*state.terminals[route_name]),
                state.route_colors.get(route_name, state.palette.text), 1,
            ))
    return guides


@dataclass(frozen=True)
class SidebarFrameState:
    auto_placement_mode: str
    show_heatmap: bool
    heatmap_scale_mode: int
    heatmap_palette_index: int
    weight_mode_index: int
    rotation_mode_index: int
    rotation_field_scores: Mapping[str, Any]
    machine_angle: float
    strategy: str
    router: str
    heuristic: str
    graph_type: str
    room_start_mode: str
    selected_route_name: str | None
    preferred_terminal_count: int
    bend_value: float
    bend_min: float
    bend_max: float
    crossing_value: float
    crossing_min: float
    crossing_max: float
    scenario_summary: Mapping[str, Any]
    fallback_frame_name: str
    machine_position: tuple[float, float]
    status: str
    validation_warnings: Sequence[str]
    elapsed_ms: float
    total_nodes: int
    fps: float


def build_sidebar_view(state: SidebarFrameState) -> SidebarView:
    heatmap = "Disabled"
    if state.show_heatmap:
        scale = "Linear" if state.heatmap_scale_mode == 0 else "Log"
        palette = "Viridis" if state.heatmap_palette_index == 1 else "Turbo"
        heatmap = f"{palette} / {scale}"
    weights = "Default" if state.weight_mode_index == 0 else "Equal (1.0)"
    rotation_mode = "Field" if state.rotation_mode_index == 1 else "Torque"
    frame = state.scenario_summary.get("routing_frame") or {}
    frame_name = str(frame.get("name") or state.fallback_frame_name).replace("_", " ")
    if state.rotation_mode_index == 1:
        scores = state.rotation_field_scores
        rotation = (
            f"Rot: {state.machine_angle}° {rotation_mode} {scores.get('selected') or '-'} "
            f"H{scores.get('H', 0.0):.3f}/V{scores.get('V', 0.0):.3f}"
        )
    else:
        rotation = f"Rotation: {state.machine_angle}° / {rotation_mode}"
    return SidebarView(
        AutoPlacementCard(state.auto_placement_mode, heatmap, weights, rotation_mode),
        SolverCard(
            state.strategy, state.router, state.heuristic, state.graph_type, state.room_start_mode,
            state.selected_route_name, state.preferred_terminal_count,
            state.bend_value, state.bend_min, state.bend_max,
            state.crossing_value, state.crossing_min, state.crossing_max,
        ),
        MachineCard(frame_name, state.machine_position, rotation),
        ExecutionStatusCard(state.status, state.validation_warnings, state.elapsed_ms, state.total_nodes, state.fps),
    )
