import sys
import os
import math
import time
import traceback
from pathlib import Path
from dataclasses import replace as _replace
import numpy as np
import pygame
from shapely.geometry import Polygon, LineString, Point
from mep_routing.domain import (
    local_axis_to_world as _local_axis_to_world,
    machine_pins as _machine_pins,
    port_access_specs as _port_access_specs,
)
from mep_routing.config import SALUBRIDAD_DEFAULTS as _SALUBRIDAD_DEFAULTS
from mep_routing.app import ActiveDwellingSession
from mep_routing.data_sources import (
    build_wall_polygons as _build_wall_polygons_for_dwelling,
    build_synthetic_dwelling as _build_synthetic_dwelling_for_layout,
    choose_initial_machine_position as _choose_initial_machine_position_for_dwelling,
    derive_room_boundary_walls as _derive_room_boundary_walls_for_dwelling,
    discover_dwelling_cases as _discover_dwelling_cases,
    prepare_real_dwelling as _prepare_real_dwelling,
    prepare_synthetic_dwelling as _prepare_synthetic_dwelling,
)
from mep_routing.installations.sal import (
    SAL_INSTALLATION,
    SalLiveRoutingSession,
    SalInteractiveCallbacks,
    SalInteractiveSolver,
    SalRouteMaterializer,
    SalSolverSettings,
)
from mep_routing.geometry import iter_polygons as _iter_polygons
from mep_routing.graphs import (
    GraphLifecycle,
)
from mep_routing.placement import (
    PlacementApplicationAdapter,
    available_machine_placement_region as _available_machine_placement_region,
    insufficient_machine_clearance_regions as _insufficient_machine_clearance_regions,
    scores_outside_regions as _scores_outside_regions,
    is_machine_placement_valid as _is_machine_placement_valid_for_placement,
    placement_weights as _placement_weights,
)
from mep_routing.routing import (
    RoutingWorkspace,
    block_terminal_node_edges as _block_terminal_node_edges,
    count_segment_crossings as _count_segment_crossings,
    count_solution_turns as _count_solution_turns,
    find_route_at_point as _find_route_at_point,
    find_route_hit_at_point as _find_route_hit_at_point,
    line_graph_dir_from_points as _line_graph_dir_from_points_for_env,
    selected_pin_names as _selected_pin_names,
    select_shaft_entry_nodes as _select_shaft_entry_nodes,
    shaft_entry_geometry as _shaft_entry_geometry_for_shaft,
    shaft_entry_segments as _shaft_entry_segments_for_geometry,
    terminal_node_indices as _terminal_node_indices_for_kd,
    terminal_validity_entries as _terminal_validity_entries,
    total_route_length_m as _total_route_length_m,
    weighted_edge_cost as _weighted_edge_cost_for_weights,
)
from mep_routing.ui.runtime import CanvasUiRuntime
from mep_routing.ui.drawing import (
    draw_dashed_polyline as _draw_dashed_polyline,
    draw_outlined_text as _draw_outlined_text,
)
from mep_routing.ui.controls import dwelling_selector_bounds as _dwelling_selector_bounds
from mep_routing.ui.solution_logs import (
    DEFAULT_BEST_METRICS,
    draw_solution_logs_panel as _draw_solution_logs_panel_widget,
    metric_value_for_log as _metric_value_for_log_entry,
    solution_log_action as _solution_log_action,
    solution_kpis as _solution_kpis,
)
from mep_routing.ui.terminal_selection import (
    draw_preferred_terminal_areas as _draw_preferred_terminal_areas,
    draw_preferred_terminal_markers as _draw_preferred_terminal_markers,
    draw_routed_terminal_endpoint_markers as _draw_routed_terminal_endpoint_markers,
)
from mep_routing.ui.terminal_validity import (
    draw_terminal_validity_overlay as _draw_terminal_validity_overlay,
    draw_terminal_validity_square as _draw_terminal_validity_square,
    draw_terminal_validity_tooltip as _draw_terminal_validity_tooltip,
)
from mep_routing.ui.overlays import (
    draw_terminal_area_drag as _draw_terminal_area_drag,
    draw_wet_room_outer_accents as _draw_wet_room_outer_accents,
)
from mep_routing.ui.help import (
    draw_card_help_button as _draw_card_help_button,
    draw_help_popup as _draw_help_popup,
    draw_transient_message as _draw_transient_message,
    draw_viewer_legend as _draw_viewer_legend,
    help_lines as _help_lines,
)
from mep_routing.observability import (
    RoutingHistory,
    SolutionLogSession,
    history_sample as _history_sample,
    restored_snapshot_state as _restored_snapshot_state,
    solution_snapshot as _solution_snapshot,
)
from mep_routing.ui.canvas import (
    CanvasFonts,
    CanvasRenderHooks,
    draw_canvas_scene as _draw_canvas_scene,
)
from mep_routing.ui.frame import (
    CanvasFrameCallbacks,
    CanvasFramePalette,
    CanvasFrameState,
    SidebarFrameState,
    active_selected_route,
    build_canvas_scene,
    build_sidebar_view,
)
from mep_routing.ui.plots import draw_routing_plots as _draw_routing_plots
from mep_routing.ui.sidebar import (
    SidebarColors,
    SidebarFonts,
    draw_sidebar as _draw_sidebar,
)
from mep_routing.ui.events import (
    CanvasGestureState as _CanvasGestureState,
    CanvasHit as _CanvasHit,
    PanelHit as _PanelHit,
    PanelInteractionState as _PanelInteractionState,
    begin_canvas_gesture as _begin_canvas_gesture,
    begin_panel_interaction as _begin_panel_interaction,
    end_canvas_gesture as _end_canvas_gesture,
    end_panel_interaction as _end_panel_interaction,
    move_canvas_gesture as _move_canvas_gesture,
    move_panel_interaction as _move_panel_interaction,
    routing_key_transition as _routing_key_transition,
)
from mep_routing.ui.installation_header import (
    InstallationHeaderState,
    InstallationOption,
    activate_installation as _activate_installation,
    draw_installation_header as _draw_installation_header,
    installation_header_hit as _installation_header_hit,
    installation_header_layout as _installation_header_layout,
    reorder_installation_at_x as _reorder_installation_at_x,
    toggle_installation as _toggle_installation,
)

# Add relative paths to sys.path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '08-bend-aware-non-orthogonal')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

DWELLING_EXPORT_ROOT = _SALUBRIDAD_DEFAULTS.get_default("DWELLING_EXPORT_ROOT")
if DWELLING_EXPORT_ROOT.exists():
    sys.path.append(str(DWELLING_EXPORT_ROOT))

import generative_layout
from solver import estimate_turns

try:
    from dwelling_export.demo_loader import load_dwelling_scenario, scenario_summary
except ImportError:
    load_dwelling_scenario = None
    scenario_summary = None

# Constants
SCALE_TO_MM    = 1000
WALL_THICKNESS = _SALUBRIDAD_DEFAULTS.get_default("WALL_THICKNESS")
GRID_SPACING   = 200    # mm ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â regular routing grid resolution
HANNAN_SCAFFOLD_SPACING = 600  # mm, static connectivity scaffold for dynamic Hannan axes
CORE_EPSILON_GRID_MM = _SALUBRIDAD_DEFAULTS.get_default("CORE_EPSILON_GRID_MM")
GRID_SPACING = _SALUBRIDAD_DEFAULTS.get_default("GRID_SPACING")
HANNAN_SCAFFOLD_SPACING = _SALUBRIDAD_DEFAULTS.get_default("HANNAN_SCAFFOLD_SPACING")
MACHINE_SPEC = SAL_INSTALLATION.default_machine
SMALL_PIN_STUB_LENGTH = MACHINE_SPEC.small_pin_stub_length_mm
LARGE_PIN_STUB_LENGTH = MACHINE_SPEC.large_pin_stub_length_mm
MACHINE_CLEARANCE = 0
MACHINE_BODY_W = MACHINE_SPEC.body_width_mm
MACHINE_BODY_H = MACHINE_SPEC.body_height_mm
MACHINE_OVERALL_W = MACHINE_SPEC.overall_width_mm
MACHINE_SMALL_DUCT_D = MACHINE_SPEC.small_duct_diameter_mm
MACHINE_LARGE_DUCT_D = MACHINE_SPEC.large_duct_diameter_mm
DUCT_BUFFER_RATIO = _SALUBRIDAD_DEFAULTS.get_default("DUCT_BUFFER_RATIO")
ROUTING_WALL_CLEARANCE_MM = _SALUBRIDAD_DEFAULTS.get_default("ROUTING_WALL_CLEARANCE_MM")
TERMINAL_REGULATION_CLEARANCE_MM = _SALUBRIDAD_DEFAULTS.get_default("TERMINAL_REGULATION_CLEARANCE_MM")
BUFFER_ROOM_TERMINALES_AIRE_MM = _SALUBRIDAD_DEFAULTS.get_default("BUFFER_ROOM_TERMINALES_AIRE_MM")
PATINEJO_CLEARANCE_MM = _SALUBRIDAD_DEFAULTS.get_default("PATINEJO_CLEARANCE_MM")
SHAFT_ENTRY_SEARCH_MM = _SALUBRIDAD_DEFAULTS.get_default("SHAFT_ENTRY_SEARCH_MM")
SHAFT_ENTRY_MAX_CANDIDATES = _SALUBRIDAD_DEFAULTS.get_default("SHAFT_ENTRY_MAX_CANDIDATES")
MACHINE_CLEARANCE_SOFT_MARGIN_MM = _SALUBRIDAD_DEFAULTS.get_default("MACHINE_CLEARANCE_SOFT_MARGIN_MM")
FPS = _SALUBRIDAD_DEFAULTS.get_default("FPS")
WHEEL_ROTATE_COOLDOWN_MS = 180
C_BEND_DEFAULT = _SALUBRIDAD_DEFAULTS.get_default("C_BEND_DEFAULT")
C_BEND_MIN = _SALUBRIDAD_DEFAULTS.get_default("C_BEND_MIN")
C_BEND_MAX = _SALUBRIDAD_DEFAULTS.get_default("C_BEND_MAX")
CROSSING_MULTIPLIER_DEFAULT = _SALUBRIDAD_DEFAULTS.get_default("CROSSING_MULTIPLIER_DEFAULT")
CROSSING_MULTIPLIER_MIN = _SALUBRIDAD_DEFAULTS.get_default("CROSSING_MULTIPLIER_MIN")
CROSSING_MULTIPLIER_MAX = _SALUBRIDAD_DEFAULTS.get_default("CROSSING_MULTIPLIER_MAX")
C_BEND         = C_BEND_DEFAULT  # Turn penalty in mm
crossing_penalty_multiplier = CROSSING_MULTIPLIER_DEFAULT
OVERLAP_BLOCK_WEIGHT = _SALUBRIDAD_DEFAULTS.get_default("OVERLAP_BLOCK_WEIGHT")
MIN_PIECE_FACTOR_DEFAULT = _SALUBRIDAD_DEFAULTS.get_default("MIN_PIECE_FACTOR_DEFAULT")
MIN_PIECE_FACTOR_MIN = _SALUBRIDAD_DEFAULTS.get_default("MIN_PIECE_FACTOR_MIN")
MIN_PIECE_FACTOR_MAX = _SALUBRIDAD_DEFAULTS.get_default("MIN_PIECE_FACTOR_MAX")
min_piece_factor = MIN_PIECE_FACTOR_DEFAULT

# Graph types
GRAPH_TYPES = SAL_INSTALLATION.graph_modes
graph_type_idx = GRAPH_TYPES.index(SAL_INSTALLATION.default_graph_mode)

ROUTING_STRATEGIES = SAL_INSTALLATION.strategy_labels
routing_strategy_idx = SAL_INSTALLATION.routing_strategies.index(
    SAL_INSTALLATION.default_routing_strategy
)

ROUTER_BACKENDS = SAL_INSTALLATION.search_backend_labels
router_backend_idx = SAL_INSTALLATION.search_backends.index(
    SAL_INSTALLATION.default_search_backend
)

HEURISTIC_MODES = SAL_INSTALLATION.heuristic_mode_labels
heuristic_mode_idx = HEURISTIC_MODES.index(SAL_INSTALLATION.default_heuristic_mode)

AUTO_PLACEMENT_MODES = [
    "Manual",
    "Topological Fields",
    "Routing-Core Workflow"
]
auto_placement_mode_idx = 2
ROTATION_MODES = [
    "Torque",
    "Field",
]
rotation_mode_idx = 0
ROTATION_FIELD_EPS = 1e-6
rotation_field_scores = {"H": 0.0, "V": 0.0, "selected": None}
show_grid_graph = False
show_heatmap = False
edge_weight_heatmap_enabled = False
edge_weight_view_mode_idx = 0
route_real_diameter_width_enabled = False
routing_workspace = RoutingWorkspace()
active_dwelling = None
help_popup_card = None
transient_message = None
transient_message_until_ms = 0
help_button_rects = {}
preferred_terminal_tool_mode = None
terminal_runtime = None
terminal_validity_overlay_enabled = False
terminal_validity_cache = {"key": None, "entries": [], "reasons_by_node": {}}
PREFERRED_TERMINAL_MARKER_SIZE_MM = 125.0
PREFERRED_TERMINAL_REMAP_TOLERANCE_MM = 300.0

ROOM_START_MODES = [
    "Room node set",
    "Centroid terminal",
]
room_start_mode_idx = 0

DWELLING_SOURCE_MODES = [
    "Real DB",
    "Random Synthetic",
]
dwelling_source_idx = 0

INSTALLATION_OPTIONS = (
    InstallationOption("sal", "Sal", available=True),
    InstallationOption("cli", "Cli"),
    InstallationOption("san", "San"),
    InstallationOption("coc", "Coc"),
)
installation_header_state = InstallationHeaderState(
    order=tuple(option.key for option in INSTALLATION_OPTIONS),
    enabled=frozenset({"sal"}),
    active="sal",
)

# Pygame Window Config
INSTALLATION_HEADER_TOP = 4
PANEL_W = 280          # right-side plot panel
COLORBAR_W = 56        # reserved lane between drawing canvas and right-side plots
canvas_ui = CanvasUiRuntime()
is_fullscreen = False
preference_reset_rect = pygame.Rect(0, 0, 1, 1)

# Color Scheme
COLOR_BG = (240, 238, 233)
COLOR_PANEL = (35, 35, 45)
COLOR_PLOT_BG = (22, 22, 30)
COLOR_ROOM = (45, 52, 54)
COLOR_ROOM_COVERED = (248, 247, 243)
COLOR_COVER_OVERLAY = (96, 104, 112, 34)
COLOR_HALLWAY = (57, 72, 85)
COLOR_WALL = (0, 0, 0)
COLOR_WET_ROOM_ACCENT = (95, 178, 218)
WALL_DRAW_WIDTH = 2
COLOR_COLUMN = (0, 0, 0)
COLOR_SHAFT = (231, 76, 60)
COLOR_SHAFT_BG = (231, 76, 60, 40)
COLOR_SHAFT_INACTIVE = (154, 84, 82)
COLOR_SHAFT_INACTIVE_HATCH = (92, 58, 58)
COLOR_MACHINE_CLEARANCE_HATCH = (105, 105, 105)
COLOR_MACHINE_CLEARANCE_FILL = (145, 145, 145, 235)
COLOR_DOOR = (220, 220, 220)
COLOR_MACHINE = (127, 140, 141)
COLOR_MACHINE_HOVER = (149, 165, 166)
COLOR_TEXT = (236, 240, 241)
COLOR_MUTED = (149, 165, 166)
COLOR_DESELECTED_ROUTE = (7, 8, 10)
COLOR_DESELECTED_ROOM = (39, 42, 43)
COLOR_DESELECTED_PIN = (112, 116, 120)
COLOR_SELECTION_HALO = (210, 235, 255)
COLOR_BLOCKED_EDGE = (82, 76, 88)
COLOR_GRAPH_EDGE = (122, 130, 140)
COLOR_GRAPH_NODE = (96, 106, 116)
COLOR_PLAN_LABEL = (18, 22, 28)
COLOR_PLAN_LABEL_HALO = (255, 255, 255)
COLOR_TERMINAL_ALLOWED = (78, 158, 178)
COLOR_TERMINAL_BLOCKED = (142, 78, 86)
COLOR_TERMINAL_BLOCKED_HATCH = (96, 54, 62)

# Route Colors
ROUTE_COLORS = {
    "Shaft": (46, 204, 113),      # Green
    "Kitchen": (241, 196, 15),    # Yellow
    "Bathroom": (52, 152, 219),   # Blue
    "Bathroom 1": (52, 152, 219), # Blue
    "Bathroom 2": (155, 89, 182), # Purple
    "Toilet": (230, 126, 34),     # Orange
    "Washroom": (26, 188, 156)    # Turquoise
}

REAL_DWELLING_DB = _SALUBRIDAD_DEFAULTS.get_default("REAL_DWELLING_DB")
PREFERRED_SHAFT_INSTALLATION = _SALUBRIDAD_DEFAULTS.get_default("PREFERRED_SHAFT_INSTALLATION")
try:
    REAL_DWELLING_SCENARIOS = _discover_dwelling_cases(REAL_DWELLING_DB)
except (ImportError, OSError) as error:
    print(f"Real dwelling catalog unavailable: {error}")
    REAL_DWELLING_SCENARIOS = ()
if not REAL_DWELLING_SCENARIOS:
    dwelling_source_idx = DWELLING_SOURCE_MODES.index("Random Synthetic")
ROUTING_FRAME_OPTIONS = [
    "dominant_walls",
    "area_inertia_allowed",
    "area_inertia_rooms",
]
real_scenario_idx = 0
routing_frame_idx = 0
current_scenario_label = "synthetic"
current_scenario_summary = {}

# Observation state for the right-side plots and solution-log panel.
HIST_MAXLEN = 400
routing_history = RoutingHistory(maxlen=HIST_MAXLEN)
solution_log_session = SolutionLogSession()
log_button_rect = pygame.Rect(0, 0, 1, 1)
log_row_rects = []
weight_mode_idx = 0                             # 0 for Default, 1 for Equal Weights
heatmap_scale_mode = 0                          # 0 for Linear (75% Saturation), 1 for Log Scale
heatmap_palette_idx = 0                         # 0 for Turbo, 1 for Viridis
to_screen = canvas_ui.viewport.to_screen
to_mm = canvas_ui.viewport.to_world


def get_installation_header_layout():
    count = len(installation_header_state.order)
    available_width = max(420, canvas_ui.viewport.canvas_width - 28)
    pill_width = min(126, max(92, (available_width - (count - 1) * 30) // count))
    gap = min(42, max(24, (available_width - count * pill_width) // max(1, count - 1)))
    return _installation_header_layout(
        canvas_ui.viewport.canvas_left,
        INSTALLATION_HEADER_TOP,
        installation_header_state.order,
        pill_width=pill_width,
        gap=gap,
    )

def handle_canvas_tool_button_click(pos):
    global route_real_diameter_width_enabled
    for action, rect, _ in canvas_ui.toolbar_buttons():
        if not rect.collidepoint(pos):
            continue
        canvas_ui.apply_view_action(action)
        if action == "diameter":
            route_real_diameter_width_enabled = not route_real_diameter_width_enabled
        return action
    if canvas_ui.weight_switch_rect().collidepoint(pos):
        return "weight_view"
    return None

def set_min_piece_factor_from_slider_x(x):
    global min_piece_factor
    min_piece_factor = canvas_ui.slider_value(
        "min_piece", x, MIN_PIECE_FACTOR_MIN, MIN_PIECE_FACTOR_MAX, min_piece_factor,
    )

def set_bend_weight_from_slider_x(x):
    global C_BEND
    raw_value = canvas_ui.slider_value("bend", x, C_BEND_MIN, C_BEND_MAX, C_BEND)
    C_BEND = float(round(raw_value / 100.0) * 100)

def set_crossing_weight_from_slider_x(x):
    global crossing_penalty_multiplier
    crossing_penalty_multiplier = canvas_ui.slider_value(
        "crossing", x, CROSSING_MULTIPLIER_MIN, CROSSING_MULTIPLIER_MAX,
        crossing_penalty_multiplier,
    )

def reset_bend_weight():
    global C_BEND
    C_BEND = C_BEND_DEFAULT

def reset_crossing_weight():
    global crossing_penalty_multiplier
    crossing_penalty_multiplier = CROSSING_MULTIPLIER_DEFAULT

def draw_min_piece_slider(screen, font_small, x, y, width):
    canvas_ui.draw_min_piece_slider(
        screen, font_small, x, y, width, min_piece_factor,
        MIN_PIECE_FACTOR_MIN, MIN_PIECE_FACTOR_MAX, COLOR_TEXT, COLOR_MUTED,
    )

def draw_weight_slider(screen, font_small, x, y, width, label, value, min_value, max_value, color, rect_name, suffix="", integer=False):
    canvas_ui.draw_weight_slider(
        screen, font_small, x, y, width, label, value, min_value, max_value,
        COLOR_TEXT, COLOR_MUTED, name=rect_name, suffix=suffix, integer=integer,
    )

def record_current_solution(routes, elapsed_ms, marker_label=None, marker_color=(241, 196, 15)):
    if routes:
        crossings_c = count_segment_crossings(routes)
        record_history(routes, crossings_c, elapsed_ms)
        if marker_label:
            routing_history.add_marker(marker_label, marker_color)

def handle_terminal_tool_button_click(pos):
    global preferred_terminal_tool_mode, terminal_validity_overlay_enabled
    for mode, rect, _ in canvas_ui.terminal_buttons():
        if not rect.collidepoint(pos):
            continue
        if mode == "reset":
            preferred_terminal_tool_mode = None
            return "reset"
        if mode == "map":
            terminal_validity_overlay_enabled = not terminal_validity_overlay_enabled
            return "map"
        preferred_terminal_tool_mode = None if preferred_terminal_tool_mode == mode else mode
        return "mode"
    return None

def draw_terminal_tool_buttons(screen, font_bold, font_small):
    canvas_ui.draw_terminal_tools(
        screen, font_bold, font_small, active_mode=preferred_terminal_tool_mode,
        overlay_enabled=terminal_validity_overlay_enabled,
        colors=(COLOR_TEXT, COLOR_MUTED, COLOR_TERMINAL_ALLOWED),
    )

def dwelling_selector_options():
    return ("New random dwelling",) + tuple(case.label for case in REAL_DWELLING_SCENARIOS)


def draw_canvas_tool_controls(screen, font_small, ruler_mode, dwelling_selector_open=False):
    return canvas_ui.draw_toolbar(screen, font_small, state={
        "ruler_enabled": ruler_mode,
        "edge_weights_enabled": edge_weight_heatmap_enabled,
        "diameter_width_enabled": route_real_diameter_width_enabled,
        "small_weight_view": edge_weight_view_mode_idx == 0,
        "dwelling_label": current_scenario_label,
        "dwelling_options": dwelling_selector_options(),
        "dwelling_selector_open": dwelling_selector_open,
        "active_terminal_mode": preferred_terminal_tool_mode,
        "text_color": COLOR_TEXT,
    })

def set_ruler_cursor(enabled):
    try:
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR if enabled else pygame.SYSTEM_CURSOR_ARROW)
    except pygame.error:
        pass

def draw_ruler_overlay(screen, font_small, start_mm, end_mm):
    return canvas_ui.draw_ruler(screen, font_small, start_mm, end_mm, COLOR_TEXT)

def invalidate_room_start_node_cache():
    if terminal_runtime is not None:
        terminal_runtime.invalidate_candidates()

def invalidate_terminal_validity_cache():
    global terminal_validity_cache
    terminal_validity_cache = {"key": None, "entries": [], "reasons_by_node": {}}

# Auto-placement cache
ap_scores = {}
ap_fields = {}
machine_vertical_clearance_blocks = ()
def get_representative_point(poly):
    centroid = poly.centroid
    if poly.contains(centroid):
        return (round(centroid.x), round(centroid.y))
    rep = poly.representative_point()
    return (round(rep.x), round(rep.y))

def _transform_source_xy_to_demo(point_xy, metadata):
    if not point_xy or metadata is None:
        return None
    x = float(point_xy[0])
    y = float(point_xy[1])
    frame = metadata.get("routing_frame") or {}
    rotation_degrees = float(frame.get("rotation_degrees", 0.0) or 0.0)
    origin = (metadata.get("metadata") or {}).get("source_rotation_origin", [0.0, 0.0])
    bounds_origin = (metadata.get("metadata") or {}).get("bounds_origin", [0.0, 0.0])

    if abs(rotation_degrees) > 1e-9:
        theta = math.radians(-rotation_degrees)
        ox = float(origin[0])
        oy = float(origin[1])
        dx = x - ox
        dy = y - oy
        x = ox + dx * math.cos(theta) - dy * math.sin(theta)
        y = oy + dx * math.sin(theta) + dy * math.cos(theta)

    return (
        round((x - float(bounds_origin[0])) * SCALE_TO_MM),
        round((y - float(bounds_origin[1])) * SCALE_TO_MM),
    )

def _build_core_shaft_entry_specs(scenario):
    metadata = scenario_summary(scenario) if scenario_summary else {"metadata": getattr(scenario, "metadata", {})}
    raw_meta = metadata.get("metadata") or {}
    candidates = raw_meta.get("shaft_candidates") or []
    candidate_idx = raw_meta.get("shaft_candidate_index")
    if candidate_idx is None or candidate_idx < 0 or candidate_idx >= len(candidates):
        candidates = [candidate for candidate in candidates if candidate.get("candidate_for_living")]
    else:
        candidates = [candidates[candidate_idx]]

    specs = []
    for candidate in candidates:
        for routing in candidate.get("routing") or []:
            centroid = _transform_source_xy_to_demo(routing.get("centroid"), metadata)
            if centroid is None:
                continue
            exit_walls = []
            for wall in routing.get("exit_walls") or []:
                if len(wall) < 2:
                    continue
                p1 = _transform_source_xy_to_demo(wall[0], metadata)
                p2 = _transform_source_xy_to_demo(wall[1], metadata)
                if p1 is not None and p2 is not None:
                    exit_walls.append((p1, p2))
            entry_points = routing.get("entry_points") or []
            if not entry_points and routing.get("entry_point"):
                entry_points = [routing.get("entry_point")]
            for entry_raw in entry_points:
                entry = _transform_source_xy_to_demo(entry_raw, metadata)
                if entry is None:
                    continue
                normal = np.array([entry[0] - centroid[0], entry[1] - centroid[1]], dtype=np.float64)
                norm = float(np.linalg.norm(normal))
                if norm < 1e-6:
                    continue
                normal /= norm
                entry_point = Point(float(entry[0]), float(entry[1]))
                matching_wall = None
                if exit_walls:
                    matching_wall = min(
                        exit_walls,
                        key=lambda wall: LineString(wall).distance(entry_point),
                    )
                    if LineString(matching_wall).distance(entry_point) > CORE_EPSILON_GRID_MM * 0.75:
                        matching_wall = None
                specs.append({
                    "centroid": (float(centroid[0]), float(centroid[1])),
                    "entry": (float(entry[0]), float(entry[1])),
                    "normal": (float(normal[0]), float(normal[1])),
                    "exit_wall": matching_wall,
                    "id": routing.get("id"),
                })
    return specs

def _shaft_entry_geometry_for_node(node_idx, env=None):
    if node_idx in shaft_entry_geometry_by_node:
        return shaft_entry_geometry_by_node[node_idx]
    if shaft_extraction is None:
        return None
    env = env or routing_workspace.env
    if env is None or node_idx is None:
        return None

    return _shaft_entry_geometry_for_shaft(shaft_extraction, env.nodes[int(node_idx)])

def get_shaft_entry_nodes(env, kd=None):
    global shaft_entry_geometry_by_node
    if shaft_extraction is None or env is None or len(env.nodes) == 0:
        return [], None
    candidates, chosen, shaft_entry_geometry_by_node = _select_shaft_entry_nodes(
        env.nodes,
        shaft_extraction,
        search_radius_mm=SHAFT_ENTRY_SEARCH_MM,
        grid_spacing_mm=GRID_SPACING,
        max_candidates=SHAFT_ENTRY_MAX_CANDIDATES,
        core_entry_specs=shaft_core_entry_specs,
        spatial_index=kd,
    )
    return candidates, chosen

def add_shaft_entry_segments(segs, first_node_idx):
    geom = _shaft_entry_geometry_for_node(first_node_idx)
    segs.extend(_shaft_entry_segments_for_geometry(geom))

def build_regular_grid():
    pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
    result = routing_workspace.build_selected(0, pins, machine_angle, terminal_runtime=terminal_runtime)
    if result is not None:
        invalidate_room_start_node_cache()
        invalidate_terminal_validity_cache()

def build_base_regular_grid():
    t0 = time.perf_counter()
    runtime = routing_workspace.build_base_regular()
    if runtime is None:
        return
    print(f"[Base Regular Grid] Built {len(runtime.nodes)} nodes in {(time.perf_counter() - t0)*1000:.1f}ms")

def update_dynamic_env(machine_poly):
    if not routing_workspace.grid_available:
        routing_workspace.apply_machine_obstacle(
            machine_poly, {}, machine_angle, clearance_mm=MACHINE_CLEARANCE,
            terminal_runtime=terminal_runtime,
        )
        invalidate_room_start_node_cache()
        return
    t0 = time.perf_counter()
    pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
    result = routing_workspace.apply_machine_obstacle(
        machine_poly, pins, machine_angle, clearance_mm=MACHINE_CLEARANCE,
        terminal_runtime=terminal_runtime,
    )
    invalidate_room_start_node_cache()
    ms = (time.perf_counter() - t0) * 1000.0
    print(f"Grid update: {ms:.1f} ms  (blocked nodes={result.blocked_node_count}, edges={result.blocked_edge_count})")

def build_hannan_grid(machine_pins=None, shift_walls=False):
    pins = machine_pins or get_machine_pins(machine_cx, machine_cy, machine_angle)
    result = routing_workspace.build_selected(
        1, pins, machine_angle, shift_hannan_walls=shift_walls, terminal_runtime=terminal_runtime,
    )
    if result is None:
        return
    invalidate_room_start_node_cache()
    invalidate_terminal_validity_cache()
    if result.variant is not None and len(result.variant.nodes):
        print(
            f"[Hannan Simple] axes={len(result.variant.axes_x)}x{len(result.variant.axes_y)} "
            f"nodes={len(result.variant.nodes)} edges={len(result.variant.edges)} in {result.elapsed_ms:.1f}ms "
            f"(axes {result.variant.axes_ms:.1f}, nodes {result.variant.nodes_ms:.1f}, edges {result.variant.edges_ms:.1f})"
        )

def build_epsilon_grid(machine_pins=None):
    pins = machine_pins or get_machine_pins(machine_cx, machine_cy, machine_angle)
    result = routing_workspace.build_selected(2, pins, machine_angle, terminal_runtime=terminal_runtime)
    if result is None:
        return
    invalidate_room_start_node_cache()
    invalidate_terminal_validity_cache()
    if result.variant is not None and len(result.variant.nodes):
        print(
            f"[Epsilon Core-like] eps={CORE_EPSILON_GRID_MM:.0f} axes={len(result.variant.axes_x)}x{len(result.variant.axes_y)} "
            f"nodes={len(result.variant.nodes)} edges={len(result.variant.edges)} in {result.elapsed_ms:.1f}ms "
            f"(axes {result.variant.axes_ms:.1f}, nodes {result.variant.nodes_ms:.1f}, edges {result.variant.edges_ms:.1f})"
        )

def build_grid(machine_pins=None):
    if graph_type_idx == 0:
        build_regular_grid()
    elif graph_type_idx == 1:
        build_hannan_grid(machine_pins=machine_pins, shift_walls=True)
    else:
        build_epsilon_grid(machine_pins=machine_pins)

# Machine representation helper
def get_machine_pins(cx, cy, angle_deg):
    return _machine_pins(MACHINE_SPEC, cx, cy, angle_deg)

# ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬
# ROUTING UTILITIES AND CONSTRAINTS
# ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬
def get_port_access_specs(global_pins, machine_angle):
    return _port_access_specs(MACHINE_SPEC, global_pins, machine_angle)

def _sal_solver_settings():
    return SalSolverSettings(
        bend_cost=C_BEND,
        crossing_penalty_multiplier=crossing_penalty_multiplier,
        duct_buffer_ratio=DUCT_BUFFER_RATIO,
        shaft_clearance_mm=PATINEJO_CLEARANCE_MM,
        machine_clearance_soft_margin_mm=MACHINE_CLEARANCE_SOFT_MARGIN_MM,
        overlap_block_weight=OVERLAP_BLOCK_WEIGHT,
        min_piece_factor=min_piece_factor,
        heuristic_mode=heuristic_mode_idx,
        routing_strategy=routing_strategy_idx,
    )


def _sal_route_analysis(policy=None):
    return _sal_live_routing_session().route_analysis(shaft_extraction, policy)


def _sal_live_routing_session():
    dwelling = active_dwelling
    return SalLiveRoutingSession(
        installation=SAL_INSTALLATION,
        machine_spec=MACHINE_SPEC,
        workspace=dwelling.workspace,
        routing_region=dwelling.routing_region,
        rooms=tuple(dwelling.rooms),
        walls=tuple(dwelling.walls),
        wall_polygons=tuple(dwelling.wall_polygons),
        shafts=tuple(dwelling.shafts),
        machine_center=(machine_cx, machine_cy),
        machine_angle=machine_angle,
        settings=_sal_solver_settings(),
        search_backend_index=router_backend_idx,
        estimate_turns=estimate_turns,
    )


def _routing_runtime(env, policy=None):
    return _sal_live_routing_session().routing_runtime(env, policy)


def get_route_diameter(route_name):
    return _sal_route_analysis().route_diameter(route_name)


def refresh_edge_weight_view_overlay(routes):
    if routing_workspace.env is None:
        return
    diameter = MACHINE_SMALL_DUCT_D if edge_weight_view_mode_idx == 0 else MACHINE_LARGE_DUCT_D
    _routing_runtime(routing_workspace.env).refresh_edge_weight_overlay(routes, diameter)

def _room_polygon_by_name(room_name):
    return None if terminal_runtime is None else terminal_runtime.room_polygon(room_name)

def _room_cover_geometry(room_name):
    return None if terminal_runtime is None else terminal_runtime.room_cover(room_name)

def _room_terminal_valid_region(room_name):
    return None if terminal_runtime is None else terminal_runtime.valid_region(room_name)

def _room_terminal_boundary_segments(room_name):
    return None if terminal_runtime is None else terminal_runtime.boundary_segments(room_name)

def get_room_candidate_start_nodes(route_name):
    return [] if terminal_runtime is None else terminal_runtime.candidate_nodes(route_name)

def get_route_start_nodes(route_name):
    return [] if terminal_runtime is None else terminal_runtime.route_start_nodes(
        route_name, use_nearest_terminal=(room_start_mode_idx == 1),
    )

def _terminal_marker_side_px():
    return max(4, int(round(PREFERRED_TERMINAL_MARKER_SIZE_MM * canvas_ui.viewport.scale)))

def find_room_candidate_node_at_world(world_pt):
    return None if terminal_runtime is None else terminal_runtime.find_candidate_at(world_pt)

def apply_preferred_terminal_point(world_pt, remove=False):
    return (False, None) if terminal_runtime is None else terminal_runtime.apply_point_preference(
        world_pt, remove=remove,
    )

def apply_preferred_terminal_area(start_world, end_world, remove=False):
    return (False, None) if terminal_runtime is None else terminal_runtime.apply_area_preference(
        start_world, end_world, remove=remove,
    )

def draw_preferred_terminal_areas(screen, selected_route_name=None):
    if routing_workspace.env is None or terminal_runtime is None:
        return
    return _draw_preferred_terminal_areas(
        screen, terminal_runtime.preferred_areas, selected_route_name, ROUTE_COLORS,
        terminal_runtime.candidate_nodes, routing_workspace.env.nodes, to_screen,
        max(3, _terminal_marker_side_px() // 2),
    )

def draw_preferred_terminal_markers(screen, selected_route_name=None, routes=None):
    if routing_workspace.env is None or terminal_runtime is None:
        return
    return _draw_preferred_terminal_markers(
        screen, terminals.keys(), terminal_runtime.preferred_points_by_room, selected_route_name, routes,
        ROUTE_COLORS, COLOR_TEXT, terminal_runtime.candidate_nodes, routing_workspace.env.nodes, to_screen,
        _terminal_marker_side_px(), PREFERRED_TERMINAL_REMAP_TOLERANCE_MM,
    )

def draw_routed_terminal_endpoint_markers(screen, routes, selected_route_name=None):
    return _draw_routed_terminal_endpoint_markers(
        screen, routes, terminals.keys(), selected_route_name, ROUTE_COLORS, COLOR_TEXT,
        to_screen, _terminal_marker_side_px(),
    )

def draw_geometry_overlay(screen, geometries, color_rgba):
    return canvas_ui.draw_geometry(screen, geometries, color_rgba)

def draw_polygon_hatch(screen, poly, color, spacing=10, dashed=False):
    return canvas_ui.draw_hatch(screen, poly, color, spacing, dashed)

def draw_dashed_polyline(screen, points, color, width=1, dash_len=8, gap_len=5):
    return _draw_dashed_polyline(screen, points, color, width, dash_len, gap_len)

def _terminal_validity_cache_key():
    return (
        id(routing_workspace.env.nodes) if routing_workspace.env is not None else None,
        len(routing_workspace.env.nodes) if routing_workspace.env is not None else 0,
        tuple(sorted(terminals.keys())),
        id(routing_region_base),
        len(rooms),
        len(walls),
        len(wall_polys),
        tuple(id(cover) for cover in covers),
    )

def get_terminal_validity_entries():
    if routing_workspace.env is None:
        return [], {}

    key = _terminal_validity_cache_key()
    if terminal_validity_cache.get("key") == key:
        return terminal_validity_cache["entries"], terminal_validity_cache["reasons_by_node"]

    entries, reasons_by_node = _terminal_validity_entries(
        routing_workspace.env.nodes,
        routing_workspace.env.adj,
        terminals.keys(),
        routing_region_base,
        _room_polygon_by_name,
        _room_terminal_valid_region,
        _room_terminal_boundary_segments,
        TERMINAL_REGULATION_CLEARANCE_MM,
        BUFFER_ROOM_TERMINALES_AIRE_MM,
    )

    terminal_validity_cache["key"] = key
    terminal_validity_cache["entries"] = entries
    terminal_validity_cache["reasons_by_node"] = reasons_by_node
    return entries, reasons_by_node

def draw_terminal_validity_square(screen, center, side, allowed):
    return _draw_terminal_validity_square(
        screen, center, side, allowed, COLOR_TERMINAL_ALLOWED, COLOR_TERMINAL_BLOCKED,
        COLOR_TERMINAL_BLOCKED_HATCH, draw_dashed_polyline,
    )

def draw_terminal_validity_overlay(screen):
    if not terminal_validity_overlay_enabled:
        return
    entries, _ = get_terminal_validity_entries()
    marker_side = max(4, min(13, int(round(70 * canvas_ui.viewport.scale))))
    return _draw_terminal_validity_overlay(
        screen, entries, to_screen, canvas_ui.viewport.canvas_rect, marker_side,
        COLOR_TERMINAL_ALLOWED, COLOR_TERMINAL_BLOCKED, COLOR_TERMINAL_BLOCKED_HATCH,
        draw_dashed_polyline,
    )

def draw_terminal_validity_tooltip(screen, font_small):
    if not terminal_validity_overlay_enabled or routing_workspace.spatial_index is None or routing_workspace.env is None:
        return
    _entries, reasons_by_node = get_terminal_validity_entries()
    return _draw_terminal_validity_tooltip(
        screen, font_small, pygame.mouse.get_pos(), canvas_ui.viewport.canvas_rect,
        to_mm, lambda world_pt: int(routing_workspace.spatial_index.query(world_pt)[1]), routing_workspace.env.nodes,
        reasons_by_node, to_screen, canvas_ui.viewport.window_size, COLOR_TEXT,
    )

def draw_wet_room_outer_accents(screen):
    return _draw_wet_room_outer_accents(screen, wet_room_outer_accents, to_screen, _iter_polygons, COLOR_WET_ROOM_ACCENT)

def draw_outlined_text(screen, font, text, pos, color, outline_color=COLOR_PLAN_LABEL_HALO):
    return _draw_outlined_text(screen, font, text, pos, color, outline_color)

def draw_terminal_area_drag(screen, start_world, end_world):
    return _draw_terminal_area_drag(screen, start_world, end_world, to_screen)

def count_segment_crossings(routes):
    return _count_segment_crossings(routes)

def count_solution_turns(routes):
    return _count_solution_turns(routes)

def count_solution_short_pieces(routes):
    return _sal_route_analysis().count_short_pieces(routes)

def find_route_at_point(routes, world_pt):
    return _find_route_at_point(routes, world_pt, max(40.0, 8.0 / canvas_ui.viewport.scale))

def find_route_hit_at_point(routes, world_pt):
    return _find_route_hit_at_point(routes, world_pt, max(40.0, 8.0 / canvas_ui.viewport.scale))

def get_route_room_polygon(route_name):
    terminal_pt = terminals.get(route_name)
    if terminal_pt is None:
        return None
    terminal_point = Point(float(terminal_pt[0]), float(terminal_pt[1]))
    containing = [
        room.polygon
        for room in rooms
        if hasattr(room, "polygon")
        and not room.polygon.is_empty
        and (room.polygon.contains(terminal_point) or room.polygon.distance(terminal_point) < 1e-7)
    ]
    if not containing:
        return None
    return min(containing, key=lambda poly: poly.area)

def find_room_route_at_point(world_pt, route_names):
    click_pt = Point(float(world_pt[0]), float(world_pt[1]))
    if "Shaft" in route_names and shaft_extraction is not None:
        if shaft_extraction.contains(click_pt) or shaft_extraction.distance(click_pt) < max(40.0, 8.0 / canvas_ui.viewport.scale):
            return "Shaft"

    candidates = []
    for route_name in route_names:
        room_poly = get_route_room_polygon(route_name)
        if room_poly is not None and (room_poly.contains(click_pt) or room_poly.distance(click_pt) < 1e-7):
            candidates.append((room_poly.area, route_name))
    if candidates:
        candidates.sort()
        return candidates[0][1]

    for room in rooms:
        if not hasattr(room, "polygon") or room.polygon.is_empty:
            continue
        room_name = getattr(room, "name", None)
        if room_name in route_names and room.polygon.contains(click_pt):
            return room_name
    return None

def get_selected_pin_names(selected_route_name, routes, global_pins):
    return _selected_pin_names(selected_route_name, routes, global_pins)

def _sal_route_materializer(route_plan):
    workspace = active_dwelling.workspace
    return SalRouteMaterializer(
        workspace.env.nodes,
        workspace.spatial_index,
        route_plan,
        add_shaft_entry_segments if shaft_extraction else None,
    )
def get_solution_score(routes, crossings, *, policy=None):
    return _sal_route_analysis(policy).score(routes, crossings)

def get_route_validation_warnings(routes):
    warnings = _sal_route_analysis().quality_warnings(routes)
    if shaft_extraction is not None and DWELLING_SOURCE_MODES[dwelling_source_idx] == "Real DB" and not shaft_core_entry_specs:
        warnings.append("missing core shaft entry metadata")
    return warnings

# ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬
# TOPOLOGICAL DISTANCE FIELDS AUTO-PLACEMENT ALGORITHMS
# ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬
def is_machine_placement_valid(cx, cy, angle):
    global_pins = get_machine_pins(cx, cy, angle)
    return _is_machine_placement_valid_for_placement(
        cx,
        cy,
        global_pins,
        routing_region_base,
        (*walls, *wall_polys),
        columns,
        shafts,
        get_machine_vertical_clearance_blocks(),
        tuple(room.polygon for room in rooms if hasattr(room, "polygon")),
    )

def get_machine_vertical_clearance_blocks():
    return machine_vertical_clearance_blocks

def get_placement_weights():
    return _placement_weights(weight_mode_idx)

def _placement_application():
    placeable_region = _available_machine_placement_region(
        routing_region_base, get_machine_vertical_clearance_blocks(),
    )
    return PlacementApplicationAdapter(
        rooms=tuple(rooms),
        terminals=dict(terminals),
        wet_room_names=tuple(wet_room_names),
        routing_region=routing_region_base,
        shaft=shaft_extraction,
        machine_area=MACHINE_OVERALL_W * MACHINE_BODY_H,
        weight_mode=weight_mode_idx,
        weights=get_placement_weights(),
        machine_pins=get_machine_pins,
        is_valid=is_machine_placement_valid,
        representative_point=get_representative_point,
        route_room_polygon=get_route_room_polygon,
        local_axis_to_world=_local_axis_to_world,
        placeable_region=placeable_region,
    )

def get_auto_placement_scores(env, shaft_boundary_nodes):
    return _placement_application().topological_scores(
        env, routing_workspace.base_spatial_index, shaft_boundary_nodes,
    )

def ensure_placement_heatmap_scores():
    global ap_scores, ap_fields
    if ap_scores or routing_workspace.base_env is None or routing_workspace.base_spatial_index is None or shaft_extraction is None:
        return
    shaft_boundary_nodes, _ = get_shaft_entry_nodes(routing_workspace.base_env, routing_workspace.base_spatial_index)
    ap_scores, ap_fields = get_auto_placement_scores(routing_workspace.base_env, shaft_boundary_nodes)

def apply_field_alignment_rotation():
    global machine_angle, rotation_field_scores
    machine_angle, rotation_field_scores, scores = _placement_application().align_rotation(
        (machine_cx, machine_cy),
        machine_angle,
        ROTATION_FIELD_EPS,
    )
    return rotation_field_scores["selected"], scores

def apply_rotation_mode_once():
    if rotation_mode_idx != 1:
        rotation_field_scores.update({"H": 0.0, "V": 0.0, "selected": "Torque"})
        return
    before = machine_angle
    apply_field_alignment_rotation()
    if machine_angle != before:
        pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
        build_grid(machine_pins=pins)

def run_auto_placement():
    global machine_cx, machine_cy, machine_angle, ap_scores, ap_fields
    if routing_workspace.base_env is None or not shaft_extraction:
        return
        
    shaft_boundary_nodes, _shaft_node_idx = get_shaft_entry_nodes(
        routing_workspace.base_env, routing_workspace.base_spatial_index,
    )
    
    outcome, elapsed_ms = _placement_application().auto_place(
        auto_placement_mode_idx,
        routing_workspace.base_env,
        routing_workspace.base_spatial_index,
        shaft_boundary_nodes,
    )
    if outcome is None:
        return
    ap_scores, ap_fields = outcome.scores, outcome.fields
    if outcome.position is None:
        print("[Auto-Placement] No feasible machine placement satisfies the active restrictions")
        return

    machine_cx, machine_cy = outcome.position
    machine_angle = outcome.rotation
    build_grid(machine_pins=get_machine_pins(machine_cx, machine_cy, machine_angle))
    if auto_placement_mode_idx == 2:
        print(f"[Core-like Machine Placement] tried {outcome.candidate_count} feasible candidates in {elapsed_ms:.1f}ms")
    else:
        print(f"[Auto-Placement] Solved position ({machine_cx}, {machine_cy}) at rotation {machine_angle} in {elapsed_ms:.2f}ms")


def _sal_interactive_solver():
    dwelling = active_dwelling
    workspace = dwelling.workspace
    callbacks = SalInteractiveCallbacks(
        update_dynamic_env=update_dynamic_env,
        build_grid=build_grid,
        shaft_entry_nodes=lambda: get_shaft_entry_nodes(workspace.env, workspace.spatial_index),
        terminal_nodes=lambda _pin_map, shaft_idx, shaft_route: _terminal_node_indices_for_kd(
            dwelling.terminal_points, shaft_idx, workspace.spatial_index, shaft_route_name=shaft_route,
        ),
        block_terminal_edges=lambda weights, terminal_map, block_weight: _block_terminal_node_edges(
            weights, workspace.env.adj, terminal_map, block_weight,
        ),
        weighted_edge_cost=_weighted_edge_cost_for_weights,
        line_graph_direction=_line_graph_dir_from_points_for_env,
        route_start_nodes=get_route_start_nodes,
        count_crossings=count_segment_crossings,
    )
    return SalInteractiveSolver(
        live_session=_sal_live_routing_session(),
        terminals=dwelling.terminal_points,
        graph_type=graph_type_idx,
        columns=tuple(dwelling.columns),
        blocked_vertical_regions=get_machine_vertical_clearance_blocks(),
        route_materializer_factory=_sal_route_materializer,
        shaft_extraction=shaft_extraction,
        callbacks=callbacks,
    )

# ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬
# MAIN SOLVER WRAPPER
# ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬
def solve_ventilation_routing():
    if "sal" not in installation_header_state.enabled:
        return [], "Routing disabled: Sal", 0.0, 0
    result = _sal_interactive_solver().solve()
    return result.routes, result.status, result.elapsed_ms, result.total_nodes

def generate_synthetic_dwelling():
    scenario = _build_synthetic_dwelling_for_layout(
        generative_layout,
        generative_layout.Room,
        get_representative_point,
        scale_to_mm=SCALE_TO_MM,
        wall_thickness_mm=WALL_THICKNESS,
    )
    _apply_prepared_dwelling(_prepare_synthetic_dwelling(scenario), auto_place=False)


def _apply_prepared_dwelling(prepared, *, auto_place):
    global rooms, columns, shafts, covers, doors, walls, wall_polys, routing_region_base, shaft_extraction, terminals, wet_room_names
    global machine_cx, machine_cy, machine_angle, _bnd_segs
    global current_scenario_label, current_scenario_summary, shaft_core_entry_specs, shaft_entry_geometry_by_node, wet_room_outer_accents
    global terminal_runtime, machine_vertical_clearance_blocks, active_dwelling, routing_workspace
    lifecycle = GraphLifecycle(
        routing_region=prepared.routing_region_base,
        wall_polygons=prepared.wall_polygons,
        covers=prepared.covers,
        columns=prepared.columns,
        shafts=prepared.shafts,
        walls=prepared.walls,
        terminals=prepared.terminals,
        shaft_extraction=prepared.shaft_extraction,
        shaft_core_entry_specs=prepared.shaft_core_entry_specs,
        grid_spacing_mm=GRID_SPACING,
        scaffold_spacing_mm=HANNAN_SCAFFOLD_SPACING,
        wall_thickness_mm=WALL_THICKNESS,
        wall_clearance_mm=ROUTING_WALL_CLEARANCE_MM,
        epsilon_mm=CORE_EPSILON_GRID_MM,
        port_access_specs=get_port_access_specs,
    )
    active_dwelling = ActiveDwellingSession.create(
        prepared,
        lifecycle,
        regulation_clearance_mm=TERMINAL_REGULATION_CLEARANCE_MM,
        terminal_buffer_mm=BUFFER_ROOM_TERMINALES_AIRE_MM,
        remap_tolerance_mm=PREFERRED_TERMINAL_REMAP_TOLERANCE_MM,
    )
    routing_workspace = active_dwelling.workspace
    terminal_runtime = active_dwelling.terminals

    # Compatibility aliases for UI code while consumers migrate to the session.
    rooms = active_dwelling.rooms
    columns = active_dwelling.columns
    shafts = active_dwelling.shafts
    covers = active_dwelling.covers
    doors = active_dwelling.doors
    walls = active_dwelling.walls
    wall_polys = active_dwelling.wall_polygons
    routing_region_base = active_dwelling.routing_region
    shaft_extraction = active_dwelling.shaft_extraction
    terminals = active_dwelling.terminal_points
    wet_room_names = active_dwelling.wet_room_names
    wet_room_outer_accents = prepared.wet_room_outer_accents
    machine_cx, machine_cy = prepared.machine_position
    machine_angle = 0
    current_scenario_label = prepared.label
    current_scenario_summary = prepared.summary
    shaft_core_entry_specs = active_dwelling.shaft_core_entry_specs
    shaft_entry_geometry_by_node = {}
    _bnd_segs = None
    machine_vertical_clearance_blocks = _insufficient_machine_clearance_regions(
        rooms, MACHINE_SPEC.installation_height_mm, MACHINE_SPEC.installation_clearance_mm,
    )
    build_base_regular_grid()
    if auto_place:
        run_auto_placement()
    pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
    build_grid(machine_pins=pins)

def _load_real_dwelling():
    if load_dwelling_scenario is None:
        raise RuntimeError("dwelling_export.demo_loader could not be imported.")
    if not REAL_DWELLING_DB.exists():
        raise RuntimeError(f"Dwelling database not found: {REAL_DWELLING_DB}")

    if not REAL_DWELLING_SCENARIOS:
        raise RuntimeError("Dwelling export database contains no cases.")
    case = REAL_DWELLING_SCENARIOS[real_scenario_idx % len(REAL_DWELLING_SCENARIOS)]
    scenario = load_dwelling_scenario(
        db_path=REAL_DWELLING_DB,
        execution=case.execution,
        dwelling_id=case.dwelling_id,
        project_guid=case.project_guid,
        scale_to_mm=True,
        frame_name=ROUTING_FRAME_OPTIONS[routing_frame_idx % len(ROUTING_FRAME_OPTIONS)],
        preferred_shaft_installation=PREFERRED_SHAFT_INSTALLATION,
    )
    return scenario, case.label, scenario_summary(scenario) if scenario_summary else {}

def generate_new_dwelling():
    invalidate_room_start_node_cache()
    if DWELLING_SOURCE_MODES[dwelling_source_idx] == "Random Synthetic":
        generate_synthetic_dwelling()
        return

    try:
        scenario, label, summary = _load_real_dwelling()
    except Exception as err:
        print(f"Real dwelling load failed, falling back to synthetic: {err}")
        generate_synthetic_dwelling()
        return
    prepared = _prepare_real_dwelling(
        scenario,
        wall_thickness_mm=WALL_THICKNESS,
        label=label,
        summary=summary,
        derive_walls=_derive_room_boundary_walls_for_dwelling,
        build_wall_polygons=_build_wall_polygons_for_dwelling,
        choose_machine_position=lambda terminals, shaft: _choose_initial_machine_position_for_dwelling(
            terminals, shaft, get_representative_point,
        ),
        build_core_entry_specs=_build_core_shaft_entry_specs,
    )
    _apply_prepared_dwelling(prepared, auto_place=True)

def draw_colorbar(screen, node_scores):
    return canvas_ui.draw_distance_colorbar(
        screen, bool(node_scores), heatmap_scale_mode, heatmap_palette_idx, COLOR_TEXT,
    )

def get_placeable_heatmap_scores(node_scores):
    if routing_workspace.base_env is None:
        return {}
    return _scores_outside_regions(
        node_scores, routing_workspace.base_env.nodes, get_machine_vertical_clearance_blocks(),
    )

def draw_distance_heatmap(screen, node_scores):
    placeable_scores = get_placeable_heatmap_scores(node_scores)
    return canvas_ui.draw_distance_heatmap(
        screen, placeable_scores, base_env=routing_workspace.base_env,
        grid_spacing=GRID_SPACING,
        blocked_regions=get_machine_vertical_clearance_blocks(),
        scale_mode=heatmap_scale_mode, palette_index=heatmap_palette_idx,
        fill_color=COLOR_MACHINE_CLEARANCE_FILL,
        hatch_color=COLOR_MACHINE_CLEARANCE_HATCH,
    )

def draw_edge_weight_heatmap(screen):
    if edge_weight_heatmap_enabled:
        return canvas_ui.draw_edge_heatmap(
            screen, routing_workspace.overlay.values, routing_workspace.env,
            OVERLAP_BLOCK_WEIGHT, COLOR_BLOCKED_EDGE,
        )

def draw_edge_weight_colorbar(screen):
    if edge_weight_heatmap_enabled:
        return canvas_ui.draw_edge_colorbar(
            screen, routing_workspace.overlay.values, OVERLAP_BLOCK_WEIGHT,
            COLOR_BLOCKED_EDGE, COLOR_TEXT,
        )

def get_route_draw_width(route_name):
    if route_real_diameter_width_enabled:
        return max(1, int(round(get_route_diameter(route_name) * canvas_ui.viewport.scale)))
    return 5 if SAL_INSTALLATION.is_large_route(route_name) else 3

def record_history(routes, crossings_count, elapsed_ms):
    """Append one solved-route observation to the app-owned history session."""
    sample = _history_sample(routes, crossings_count, elapsed_ms, _total_route_length_m, count_solution_turns, get_solution_score)
    routing_history.append(sample)
    if routes:
        update_auto_best_logs(routes, "Auto best", elapsed_ms, 0)


def clear_history_buffers():
    routing_history.clear()

def _routes_total_length_m(routes):
    return _total_route_length_m(routes)

def get_current_kpis(routes, elapsed_ms):
    return _solution_kpis(
        routes,
        elapsed_ms,
        count_segment_crossings,
        _routes_total_length_m,
        count_solution_turns,
        count_solution_short_pieces,
        get_solution_score,
    )

def snapshot_current_state(routes, status, elapsed_ms, total_nodes):
    kpis = get_current_kpis(routes, elapsed_ms)
    return _solution_snapshot({
        "source_mode_idx": dwelling_source_idx,
        "real_scenario_idx": real_scenario_idx,
        "routing_frame_idx": routing_frame_idx,
        "scenario_label": current_scenario_label,
        "machine": (float(machine_cx), float(machine_cy), int(machine_angle)),
        "graph_type_idx": graph_type_idx,
        "routing_strategy_idx": routing_strategy_idx,
        "router_backend_idx": router_backend_idx,
        "heuristic_mode_idx": heuristic_mode_idx,
        "rotation_mode_idx": rotation_mode_idx,
        "room_start_mode_idx": room_start_mode_idx,
        "weight_mode_idx": weight_mode_idx,
        "edge_weight_view_mode_idx": edge_weight_view_mode_idx,
        "route_real_diameter_width_enabled": bool(route_real_diameter_width_enabled),
        "min_piece_factor": float(min_piece_factor),
        "bend_weight": float(C_BEND),
        "crossing_penalty_multiplier": float(crossing_penalty_multiplier),
        "preferred_terminal_points_by_room": (
            {} if terminal_runtime is None else terminal_runtime.preferred_points_by_room
        ),
        "preferred_terminal_areas": [] if terminal_runtime is None else terminal_runtime.preferred_areas,
    }, kpis, status, total_nodes)

def log_current_solution(routes, status, elapsed_ms, total_nodes):
    if not routes:
        return False
    record_history(routes, count_segment_crossings(routes), elapsed_ms)
    entry = solution_log_session.add_manual(
        snapshot_current_state(routes, status, elapsed_ms, total_nodes),
        routing_history.latest_index,
    )
    routing_history.add_marker(f"L{entry['id']}", (255, 255, 255))
    return True

def _metric_value_for_log(entry, metric):
    return _metric_value_for_log_entry(entry, metric)

def update_auto_best_logs(routes, status, elapsed_ms, total_nodes):
    if not routes:
        return
    entry_base = snapshot_current_state(routes, status, elapsed_ms, total_nodes)
    for _metric, _entry, label, color in solution_log_session.update_auto_bests(
        entry_base,
        routing_history.latest_index,
        DEFAULT_BEST_METRICS,
        _metric_value_for_log_entry,
    ):
        routing_history.replace_marker(label, routing_history.latest_index, color)

def restore_solution_log(log_entry):
    global machine_cx, machine_cy, machine_angle
    global graph_type_idx, routing_strategy_idx, router_backend_idx, heuristic_mode_idx, room_start_mode_idx
    global rotation_mode_idx
    global weight_mode_idx, edge_weight_view_mode_idx, route_real_diameter_width_enabled, min_piece_factor
    global C_BEND, crossing_penalty_multiplier

    state = _restored_snapshot_state(log_entry, C_BEND_DEFAULT, CROSSING_MULTIPLIER_DEFAULT)
    machine_cx, machine_cy, machine_angle = state["machine"]
    graph_type_idx, routing_strategy_idx, router_backend_idx = state["graph_type_idx"], state["routing_strategy_idx"], state["router_backend_idx"]
    heuristic_mode_idx, rotation_mode_idx, room_start_mode_idx = state["heuristic_mode_idx"], state["rotation_mode_idx"], state["room_start_mode_idx"]
    weight_mode_idx, edge_weight_view_mode_idx = state["weight_mode_idx"], state["edge_weight_view_mode_idx"]
    route_real_diameter_width_enabled, min_piece_factor = state["route_real_diameter_width_enabled"], state["min_piece_factor"]
    C_BEND, crossing_penalty_multiplier = state["bend_weight"], state["crossing_penalty_multiplier"]
    if terminal_runtime is not None:
        terminal_runtime.restore_preferences(
            state["preferred_terminal_points_by_room"], state["preferred_terminal_areas"],
        )
    solution_log_session.selected_log_id = state["id"]
    pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
    build_grid(machine_pins=pins)
    return solve_ventilation_routing()

def handle_solution_log_click(pos):
    return _solution_log_action(pos, log_button_rect, log_row_rects)

def draw_solution_logs_panel(screen, font_small, font_bold):
    global log_button_rect, log_row_rects
    log_button_rect, log_row_rects = _draw_solution_logs_panel_widget(
        screen,
        font_small,
        font_bold,
        window_width=canvas_ui.viewport.window_width,
        window_height=canvas_ui.viewport.window_height,
        panel_width=PANEL_W,
        solution_logs=solution_log_session.manual_logs,
        auto_best_logs=solution_log_session.auto_best_logs,
        selected_log_id=solution_log_session.selected_log_id,
        plot_background_color=COLOR_PLOT_BG,
        text_color=COLOR_TEXT,
        muted_color=COLOR_MUTED,
    )

def draw_plots(screen, font_plot_small, font_plot_title, font_plot_value, font_plot_minimum):
    return _draw_routing_plots(
        screen, font_plot_small, font_plot_title, font_plot_value, font_plot_minimum,
        canvas_ui.viewport.window_width, PANEL_W,
        routing_history.buffers,
        routing_history.sample_count, routing_history.event_markers, COLOR_PLOT_BG, COLOR_TEXT, COLOR_MUTED,
    )

def draw_card_help_button(screen, card_id, rect, font_small):
    global help_button_rects
    help_button_rects[card_id] = _draw_card_help_button(screen, card_id, rect, font_small, help_popup_card == card_id, COLOR_MUTED, COLOR_TEXT)

def draw_help_popup(screen, font_small):
    viewport = canvas_ui.viewport
    return _draw_help_popup(screen, font_small, _help_lines(help_popup_card), (viewport.canvas_left + 16, viewport.canvas_top + 58), COLOR_TEXT)

def set_transient_message(text, duration_ms=2400):
    global transient_message, transient_message_until_ms
    transient_message = str(text)
    transient_message_until_ms = pygame.time.get_ticks() + int(duration_ms)

def draw_transient_message(screen, font_small):
    if not transient_message or pygame.time.get_ticks() > transient_message_until_ms:
        return
    return _draw_transient_message(
        screen, font_small, transient_message,
        (canvas_ui.viewport.canvas_left, canvas_ui.viewport.canvas_top), COLOR_TEXT,
    )

def draw_viewer_legend(screen, font_small):
    return _draw_viewer_legend(
        screen, font_small, canvas_ui.viewport.canvas_rect, terminal_validity_overlay_enabled,
        COLOR_PLAN_LABEL, COLOR_WET_ROOM_ACCENT, COLOR_WALL, draw_terminal_validity_square,
    )


def apply_routing_key_command(event_key):
    """Apply a pure routing-key transition through the live Pygame adapters."""
    global machine_angle, auto_placement_mode_idx, routing_strategy_idx, room_start_mode_idx
    global router_backend_idx, heuristic_mode_idx, graph_type_idx, rotation_mode_idx, weight_mode_idx
    global ap_scores, ap_fields

    command = {
        pygame.K_r: "rotate_machine",
        pygame.K_c: "cycle_strategy",
        pygame.K_t: "cycle_room_start",
        pygame.K_l: "cycle_router_backend",
        pygame.K_y: "cycle_heuristic",
        pygame.K_TAB: "cycle_graph",
        pygame.K_a: "toggle_auto_placement",
        pygame.K_p: "cycle_auto_placement",
        pygame.K_u: "cycle_rotation_mode",
        pygame.K_w: "cycle_weight_mode",
    }.get(event_key)
    if command is None:
        return None

    transition = _routing_key_transition(
        command,
        {
            "machine_angle": machine_angle,
            "auto_placement_mode_idx": auto_placement_mode_idx,
            "routing_strategy_idx": routing_strategy_idx,
            "room_start_mode_idx": room_start_mode_idx,
            "router_backend_idx": router_backend_idx,
            "heuristic_mode_idx": heuristic_mode_idx,
            "graph_type_idx": graph_type_idx,
            "rotation_mode_idx": rotation_mode_idx,
            "weight_mode_idx": weight_mode_idx,
        },
        {
            "routing_strategy": len(ROUTING_STRATEGIES),
            "room_start": len(ROOM_START_MODES),
            "router_backend": len(ROUTER_BACKENDS),
            "heuristic": len(HEURISTIC_MODES),
            "graph": len(GRAPH_TYPES),
            "rotation_mode": len(ROTATION_MODES),
            "auto_placement": len(AUTO_PLACEMENT_MODES),
            "weight_mode": 2,
        },
    )
    if transition is None:
        return None

    state = transition.state
    machine_angle = state["machine_angle"]
    auto_placement_mode_idx = state["auto_placement_mode_idx"]
    routing_strategy_idx = state["routing_strategy_idx"]
    room_start_mode_idx = state["room_start_mode_idx"]
    router_backend_idx = state["router_backend_idx"]
    heuristic_mode_idx = state["heuristic_mode_idx"]
    graph_type_idx = state["graph_type_idx"]
    rotation_mode_idx = state["rotation_mode_idx"]
    weight_mode_idx = state["weight_mode_idx"]

    if transition.rebuild_graph:
        build_grid(machine_pins=get_machine_pins(machine_cx, machine_cy, machine_angle))
    if transition.apply_rotation_mode:
        apply_rotation_mode_once()
    if transition.refresh_placement_fields and routing_workspace.base_env is not None and shaft_extraction is not None:
        shaft_boundary_nodes, _ = get_shaft_entry_nodes(
            routing_workspace.base_env, routing_workspace.base_spatial_index,
        )
        ap_scores, ap_fields = get_auto_placement_scores(routing_workspace.base_env, shaft_boundary_nodes)
    return transition

def main():
    global machine_cx, machine_cy, machine_angle, show_grid_graph, graph_type_idx, routing_strategy_idx
    global router_backend_idx, heuristic_mode_idx, auto_placement_mode_idx, show_heatmap, weight_mode_idx, ap_scores, ap_fields, heatmap_scale_mode, heatmap_palette_idx
    global real_scenario_idx, routing_frame_idx, dwelling_source_idx, room_start_mode_idx
    global edge_weight_heatmap_enabled, edge_weight_view_mode_idx, route_real_diameter_width_enabled
    global rotation_mode_idx
    global help_popup_card
    global min_piece_factor, is_fullscreen
    global preferred_terminal_tool_mode
    global installation_header_state
    
    pygame.init()
    pygame.font.init()
    
    viewport = canvas_ui.viewport
    canvas_ui.update_layout(viewport.window_width, viewport.window_height)
    screen = pygame.display.set_mode(viewport.window_size, pygame.RESIZABLE)
    pygame.display.set_caption("Integrated Auto-Placement & Ventilation Router (Demo 10.8)")
    clock = pygame.time.Clock()
    
    font_title = pygame.font.SysFont("Outfit", 24, bold=True)
    font_bold = pygame.font.SysFont("Outfit", 18, bold=True)
    font_small = pygame.font.SysFont("Outfit", 15)
    font_micro = pygame.font.SysFont("Outfit", 11, bold=True)
    plot_font_family = "Arial, Liberation Sans, DejaVu Sans"
    font_plot_small = pygame.font.SysFont(plot_font_family, 11)
    font_plot_title = pygame.font.SysFont(plot_font_family, 16, bold=True)
    font_plot_value = pygame.font.SysFont(plot_font_family, 14, bold=True)
    font_plot_minimum = pygame.font.SysFont(plot_font_family, 13)
    
    generate_new_dwelling()
    
    dragging = False
    drag_offset_x = 0.0
    drag_offset_y = 0.0
    ruler_mode = False
    ruler_dragging = False
    ruler_start_mm = None
    ruler_end_mm = None
    terminal_area_dragging = False
    terminal_area_start_mm = None
    terminal_area_end_mm = None
    terminal_area_remove = False
    panning_view = False
    pan_last_pos = None
    dragging_min_piece_slider = False
    dragging_bend_weight_slider = False
    dragging_crossing_weight_slider = False
    selected_route_name = None
    dwelling_selector_open = False
    last_wheel_rotate_ms = 0
    installation_drag_key = None
    installation_drag_moved = False
    canvas_gesture = _CanvasGestureState()
    panel_interaction = _PanelInteractionState()

    def current_canvas_gesture():
        return _replace(
            canvas_gesture,
            ruler_mode=ruler_mode,
            terminal_tool_mode=preferred_terminal_tool_mode,
        )

    def apply_canvas_gesture(state):
        nonlocal canvas_gesture, dragging, drag_offset_x, drag_offset_y
        nonlocal ruler_dragging, ruler_start_mm, ruler_end_mm
        nonlocal terminal_area_dragging, terminal_area_start_mm, terminal_area_end_mm, terminal_area_remove
        nonlocal panning_view, pan_last_pos
        canvas_gesture = state
        dragging = state.machine_dragging
        drag_offset_x, drag_offset_y = state.machine_drag_offset_mm or (0.0, 0.0)
        ruler_dragging = state.ruler_dragging
        ruler_start_mm, ruler_end_mm = state.ruler_start_mm, state.ruler_end_mm
        terminal_area_dragging = state.terminal_area_dragging
        terminal_area_start_mm, terminal_area_end_mm = state.terminal_area_start_mm, state.terminal_area_end_mm
        terminal_area_remove = state.terminal_area_remove
        panning_view, pan_last_pos = state.panning, state.pan_last_screen

    def apply_panel_interaction(state):
        nonlocal panel_interaction, dragging_min_piece_slider
        nonlocal dragging_bend_weight_slider, dragging_crossing_weight_slider
        panel_interaction = state
        dragging_min_piece_slider = state.active_slider == "min_piece"
        dragging_bend_weight_slider = state.active_slider == "bend"
        dragging_crossing_weight_slider = state.active_slider == "crossing"
    
    routes = []
    status = "Initial"
    elapsed_ms = 0.0
    total_nodes = 0
    
    needs_auto_placement = (auto_placement_mode_idx > 0)
    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
    
    running = True
    while running:
        if needs_auto_placement:
            needs_auto_placement = False
            run_auto_placement()
            routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
            if routes and not status.startswith("Blocked"):
                crossings_c = count_segment_crossings(routes)
                record_history(routes, crossings_c, elapsed_ms)
                routing_history.add_marker("Auto", (230, 126, 34))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE and not is_fullscreen:
                canvas_ui.update_layout(event.w, event.h)
                screen = pygame.display.set_mode(viewport.window_size, pygame.RESIZABLE)
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    installation_hit = _installation_header_hit(
                        get_installation_header_layout(), (mx, my),
                    )
                    if installation_hit is not None:
                        hit_kind, installation_key = installation_hit
                        if hit_kind == "switch":
                            installation_header_state, changed = _toggle_installation(
                                installation_header_state, installation_key, INSTALLATION_OPTIONS,
                            )
                            if changed:
                                routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                                selected_route_name = None
                            else:
                                set_transient_message(f"{installation_key.title()} is not available yet")
                        else:
                            installation_drag_key = installation_key
                            installation_drag_moved = False
                        continue
                    selector_bounds, option_bounds = _dwelling_selector_bounds(
                        viewport.canvas_left, viewport.canvas_top, len(dwelling_selector_options())
                    )
                    if pygame.Rect(selector_bounds).collidepoint((mx, my)):
                        dwelling_selector_open = not dwelling_selector_open
                        continue
                    selected_dwelling = None
                    if dwelling_selector_open:
                        selected_dwelling = next(
                            (index for index, bounds in enumerate(option_bounds) if pygame.Rect(bounds).collidepoint((mx, my))),
                            None,
                        )
                        dwelling_selector_open = False
                    if selected_dwelling is not None:
                        if selected_dwelling == 0:
                            dwelling_source_idx = DWELLING_SOURCE_MODES.index("Random Synthetic")
                        else:
                            dwelling_source_idx = DWELLING_SOURCE_MODES.index("Real DB")
                            real_scenario_idx = selected_dwelling - 1
                        generate_new_dwelling()
                        solution_log_session.clear()
                        clear_history_buffers()
                        needs_auto_placement = auto_placement_mode_idx > 0
                        routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                        if routes and not status.startswith("Blocked"):
                            record_current_solution(routes, elapsed_ms, f"Home:{selected_dwelling}", (52, 152, 219))
                        continue
                    help_card = next(
                        (card_id for card_id, rect in help_button_rects.items() if rect.collidepoint((mx, my))),
                        None,
                    )
                    min_piece_hit = canvas_ui.controls.min_piece.collidepoint((mx, my))
                    bend_slider_hit = canvas_ui.controls.bend.collidepoint((mx, my))
                    bend_reset_hit = canvas_ui.controls.bend_reset.collidepoint((mx, my))
                    crossing_slider_hit = canvas_ui.controls.crossing.collidepoint((mx, my))
                    crossing_reset_hit = canvas_ui.controls.crossing_reset.collidepoint((mx, my))
                    panel_target_found = any((
                        help_card is not None,
                        min_piece_hit,
                        bend_slider_hit,
                        bend_reset_hit,
                        crossing_slider_hit,
                        crossing_reset_hit,
                    ))
                    tool_action = None
                    log_action = None
                    terminal_tool_action = None
                    if not panel_target_found:
                        tool_action = handle_canvas_tool_button_click((mx, my))
                        panel_target_found = tool_action is not None
                    if not panel_target_found:
                        log_action = handle_solution_log_click((mx, my))
                        panel_target_found = log_action is not None
                    if not panel_target_found:
                        terminal_tool_action = handle_terminal_tool_button_click((mx, my))
                    panel_transition = _begin_panel_interaction(
                        panel_interaction,
                        hit=_PanelHit(
                            help_card=help_card,
                            min_piece_slider=min_piece_hit,
                            bend_slider=bend_slider_hit,
                            bend_reset=bend_reset_hit,
                            crossing_slider=crossing_slider_hit,
                            crossing_reset=crossing_reset_hit,
                            canvas_tool=tool_action,
                            solution_log_action=log_action,
                            terminal_tool_action=terminal_tool_action,
                        ),
                        screen_x=mx,
                    )
                    apply_panel_interaction(panel_transition.state)
                    if panel_transition.commands:
                        panel_command = panel_transition.commands[0]
                        if panel_command.name == "toggle_help":
                            card_id = panel_command.value
                            help_popup_card = None if help_popup_card == card_id else card_id
                        elif panel_command.name == "set_slider":
                            slider_name, slider_x = panel_command.value
                            if slider_name == "min_piece":
                                set_min_piece_factor_from_slider_x(slider_x)
                                marker_label, marker_color = f"Min:{min_piece_factor:.2f}", (241, 196, 15)
                            elif slider_name == "bend":
                                set_bend_weight_from_slider_x(slider_x)
                                marker_label, marker_color = f"B:{C_BEND:.0f}", (155, 89, 182)
                            else:
                                set_crossing_weight_from_slider_x(slider_x)
                                marker_label, marker_color = f"X:{crossing_penalty_multiplier:.1f}", (230, 126, 34)
                            routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                            if routes and not status.startswith("Blocked"):
                                record_current_solution(routes, elapsed_ms, marker_label, marker_color)
                        elif panel_command.name == "reset_slider":
                            if panel_command.value == "bend":
                                reset_bend_weight()
                                marker_label, marker_color = "B:reset", (155, 89, 182)
                            else:
                                reset_crossing_weight()
                                marker_label, marker_color = "X:reset", (230, 126, 34)
                            routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                            if routes and not status.startswith("Blocked"):
                                record_current_solution(routes, elapsed_ms, marker_label, marker_color)
                        elif panel_command.name == "canvas_tool":
                            if panel_command.value == "ruler":
                                ruler_mode = not ruler_mode
                                apply_canvas_gesture(_replace(current_canvas_gesture(), ruler_dragging=False))
                                if ruler_mode:
                                    preferred_terminal_tool_mode = None
                                set_ruler_cursor(ruler_mode)
                            elif panel_command.value == "weights":
                                edge_weight_heatmap_enabled = not edge_weight_heatmap_enabled
                            elif panel_command.value == "weight_view":
                                edge_weight_view_mode_idx = (edge_weight_view_mode_idx + 1) % 2
                        elif panel_command.name == "solution_log":
                            log_action = panel_command.value
                            if log_action == "log":
                                log_current_solution(routes, status, elapsed_ms, total_nodes)
                            else:
                                if isinstance(log_action, str) and log_action.startswith("best:"):
                                    log_entry = solution_log_session.auto_best_logs.get(log_action.split(":", 1)[1])
                                else:
                                    log_entry = next((entry for entry in solution_log_session.manual_logs if entry["id"] == log_action), None)
                                if log_entry is not None:
                                    routes, status, elapsed_ms, total_nodes = restore_solution_log(log_entry)
                                    if routes and not status.startswith("Blocked"):
                                        record_current_solution(routes, elapsed_ms, f"Back:L{log_action}", (255, 255, 255))
                        elif panel_command.name == "terminal_tool":
                            if panel_command.value == "reset":
                                if terminal_runtime is not None:
                                    terminal_runtime.clear_preferences()
                                routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                                if routes and not status.startswith("Blocked"):
                                    record_current_solution(routes, elapsed_ms, "Prefs reset", (26, 188, 156))
                            if preferred_terminal_tool_mode:
                                ruler_mode = False
                                apply_canvas_gesture(_replace(current_canvas_gesture(), ruler_dragging=False))
                            set_ruler_cursor(bool(preferred_terminal_tool_mode))
                        continue
                    world_x, world_y = to_mm(mx, my)
                    mods = pygame.key.get_mods()
                    shift_pressed = bool(mods & pygame.KMOD_SHIFT)
                    ctrl_pressed = bool(mods & pygame.KMOD_CTRL)
                    canvas_hit = _CanvasHit()
                    machine_center = None
                    if not shift_pressed and not ruler_mode and preferred_terminal_tool_mode is None:
                        g_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
                        m_poly = Polygon([g_pins["c_tl"], g_pins["c_tr"], g_pins["c_br"], g_pins["c_bl"]])
                        p_obj = Point(world_x, world_y)
                        machine_hit = m_poly.contains(p_obj) or m_poly.distance(p_obj) < 200.0
                        route_names = {name for name, _ in routes} if routes else set()
                        room_hit = find_room_route_at_point((world_x, world_y), route_names)
                        route_hit = find_route_hit_at_point(routes, (world_x, world_y))
                        direct_duct_click_mm = max(20.0, 4.0 / viewport.scale)
                        direct_route_name = route_hit[0] if route_hit and (not room_hit or route_hit[1] <= direct_duct_click_mm) else None
                        canvas_hit = _CanvasHit(
                            machine_hit=machine_hit,
                            route_name=direct_route_name,
                            room_route_name=room_hit,
                        )
                        machine_center = (machine_cx, machine_cy) if machine_hit else None
                    canvas_transition = _begin_canvas_gesture(
                        current_canvas_gesture(),
                        world_point=(world_x, world_y),
                        screen_point=(mx, my),
                        shift=shift_pressed,
                        ctrl=ctrl_pressed,
                        hit=canvas_hit,
                        machine_center_mm=machine_center,
                    )
                    apply_canvas_gesture(canvas_transition.state)
                    for canvas_command in canvas_transition.commands:
                        if canvas_command.name == "start_pan":
                            try:
                                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEALL)
                            except pygame.error:
                                pass
                        elif canvas_command.name == "apply_terminal_point":
                            terminal_point, remove_marker = canvas_command.value
                            changed, marker_room = apply_preferred_terminal_point(terminal_point, remove=remove_marker)
                            if marker_room:
                                selected_route_name = marker_room
                            elif not remove_marker:
                                set_transient_message("Invalid terminal: too close to wall or outside allowed room buffer")
                            if changed:
                                routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                                if routes and not status.startswith("Blocked"):
                                    label = "Term-" if remove_marker else "Term+"
                                    record_current_solution(routes, elapsed_ms, label, (26, 188, 156))
                        elif canvas_command.name == "start_machine_drag":
                            auto_placement_mode_idx = 0
                        elif canvas_command.name == "select_route":
                            selected_route_name = canvas_command.value
                        elif canvas_command.name == "clear_route_selection":
                            selected_route_name = None
                    continue
                elif event.button == 2:
                    mouse_x, mouse_y = event.pos
                    canvas_transition = _begin_canvas_gesture(
                        current_canvas_gesture(),
                        world_point=to_mm(mouse_x, mouse_y),
                        screen_point=(mouse_x, mouse_y),
                        shift=True,
                        ctrl=False,
                        hit=_CanvasHit(),
                    )
                    apply_canvas_gesture(canvas_transition.state)
                    try:
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEALL)
                    except pygame.error:
                        pass
                elif event.button == 4: # Scroll Up (CCW)
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        canvas_ui.zoom_at(viewport.zoom * 1.12, event.pos)
                        continue
                    now_ms = pygame.time.get_ticks()
                    if now_ms - last_wheel_rotate_ms < WHEEL_ROTATE_COOLDOWN_MS:
                        continue
                    last_wheel_rotate_ms = now_ms
                    auto_placement_mode_idx = 0
                    machine_angle = (machine_angle + 90) % 360
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        routing_history.add_marker(f"Rot:{machine_angle}", (46, 204, 113))
                elif event.button == 5: # Scroll Down (CW)
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        canvas_ui.zoom_at(viewport.zoom / 1.12, event.pos)
                        continue
                    now_ms = pygame.time.get_ticks()
                    if now_ms - last_wheel_rotate_ms < WHEEL_ROTATE_COOLDOWN_MS:
                        continue
                    last_wheel_rotate_ms = now_ms
                    auto_placement_mode_idx = 0
                    machine_angle = (machine_angle - 90) % 360
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        routing_history.add_marker(f"Rot:{machine_angle}", (46, 204, 113))
                        
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 2):
                    if event.button == 1 and installation_drag_key is not None:
                        if not installation_drag_moved:
                            installation_header_state, changed = _activate_installation(
                                installation_header_state, installation_drag_key, INSTALLATION_OPTIONS,
                            )
                            if not changed and installation_drag_key != installation_header_state.active:
                                set_transient_message(f"{installation_drag_key.title()} is not available yet")
                        installation_drag_key = None
                        installation_drag_moved = False
                        continue
                    apply_panel_interaction(_end_panel_interaction(panel_interaction).state)
                    canvas_transition = _end_canvas_gesture(
                        current_canvas_gesture(),
                        button="left" if event.button == 1 else "middle",
                    )
                    apply_canvas_gesture(canvas_transition.state)
                    for canvas_command in canvas_transition.commands:
                        if canvas_command.name != "apply_terminal_area":
                            continue
                        start_point, end_point, remove_area = canvas_command.value
                        changed, marker_room = apply_preferred_terminal_area(start_point, end_point, remove=remove_area)
                        if marker_room:
                            selected_route_name = marker_room
                        if changed:
                            routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                            if routes and not status.startswith("Blocked"):
                                label = "Area-" if remove_area else "Area+"
                                record_current_solution(routes, elapsed_ms, label, (155, 89, 182))
                    set_ruler_cursor(ruler_mode or bool(preferred_terminal_tool_mode))
                    
            elif event.type == pygame.MOUSEMOTION:
                if installation_drag_key is not None:
                    reordered = _reorder_installation_at_x(
                        installation_header_state,
                        installation_drag_key,
                        event.pos[0],
                        get_installation_header_layout(),
                    )
                    if reordered.order != installation_header_state.order:
                        installation_header_state = reordered
                        installation_drag_moved = True
                    continue
                panel_transition = _move_panel_interaction(panel_interaction, screen_x=event.pos[0])
                apply_panel_interaction(panel_transition.state)
                if panel_transition.commands:
                    slider_name, slider_x = panel_transition.commands[0].value
                    if slider_name == "min_piece":
                        set_min_piece_factor_from_slider_x(slider_x)
                    elif slider_name == "bend":
                        set_bend_weight_from_slider_x(slider_x)
                    else:
                        set_crossing_weight_from_slider_x(slider_x)
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        record_current_solution(routes, elapsed_ms)
                    continue
                mouse_x, mouse_y = event.pos
                canvas_transition = _move_canvas_gesture(
                    current_canvas_gesture(),
                    world_point=to_mm(mouse_x, mouse_y),
                    screen_point=(mouse_x, mouse_y),
                )
                apply_canvas_gesture(canvas_transition.state)
                for canvas_command in canvas_transition.commands:
                    if canvas_command.name == "pan_by":
                        dx, dy = canvas_command.value
                        canvas_ui.pan_by(dx, dy)
                    elif canvas_command.name == "move_machine":
                        machine_cx, machine_cy = canvas_command.value
                        routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                        if routes and not status.startswith("Blocked"):
                            crossings_c = count_segment_crossings(routes)
                            record_history(routes, crossings_c, elapsed_ms)
                    
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        info = pygame.display.Info()
                        screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
                        canvas_ui.update_layout(screen.get_width(), screen.get_height())
                    else:
                        canvas_ui.update_layout(1700, 930)
                        screen = pygame.display.set_mode(viewport.window_size, pygame.RESIZABLE)

                elif event.key == pygame.K_ESCAPE:
                    if ruler_mode:
                        ruler_mode = False
                        apply_canvas_gesture(_replace(current_canvas_gesture(), ruler_dragging=False))
                        set_ruler_cursor(False)
                    elif preferred_terminal_tool_mode:
                        preferred_terminal_tool_mode = None
                        apply_canvas_gesture(_replace(current_canvas_gesture(), terminal_area_dragging=False))
                        set_ruler_cursor(False)
                    else:
                        selected_route_name = None
                elif event.key == pygame.K_SPACE:
                    if DWELLING_SOURCE_MODES[dwelling_source_idx] == "Real DB":
                        real_scenario_idx = (real_scenario_idx + 1) % len(REAL_DWELLING_SCENARIOS)
                    generate_new_dwelling()
                    solution_log_session.clear()
                    clear_history_buffers()
                    if auto_placement_mode_idx > 0:
                        needs_auto_placement = True
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                    
                elif routing_transition := apply_routing_key_command(event.key):
                    if routing_transition.needs_auto_placement:
                        needs_auto_placement = True
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routing_transition.record_history and routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        for marker_label, marker_color in routing_transition.markers:
                            routing_history.add_marker(marker_label, marker_color)
                    
                elif event.key == pygame.K_g:
                    show_grid_graph = not show_grid_graph
                    
                elif event.key == pygame.K_d:
                    dwelling_source_idx = (dwelling_source_idx + 1) % len(DWELLING_SOURCE_MODES)
                    generate_new_dwelling()
                    solution_log_session.clear()
                    clear_history_buffers()
                    if auto_placement_mode_idx > 0:
                        needs_auto_placement = True
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        routing_history.add_marker(f"Src:{dwelling_source_idx}", (52, 152, 219))

                elif event.key == pygame.K_o:
                    routing_frame_idx = (routing_frame_idx + 1) % len(ROUTING_FRAME_OPTIONS)
                    generate_new_dwelling()
                    solution_log_session.clear()
                    if auto_placement_mode_idx > 0:
                        needs_auto_placement = True
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        routing_history.add_marker(f"Frame:{routing_frame_idx}", (230, 126, 34))

                elif event.key == pygame.K_v:
                    show_heatmap = not show_heatmap
                    if show_heatmap:
                        ensure_placement_heatmap_scores()

                elif event.key == pygame.K_m:
                    edge_weight_heatmap_enabled = not edge_weight_heatmap_enabled

                elif event.key == pygame.K_n:
                    edge_weight_view_mode_idx = (edge_weight_view_mode_idx + 1) % 2

                elif event.key == pygame.K_x:
                    route_real_diameter_width_enabled = not route_real_diameter_width_enabled
                    
                elif event.key == pygame.K_h:
                    heatmap_scale_mode = (heatmap_scale_mode + 1) % 2
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        routing_history.add_marker(f"H:{'Log' if heatmap_scale_mode==1 else 'Lin'}", (150, 150, 150))

                elif event.key == pygame.K_b:
                    heatmap_palette_idx = (heatmap_palette_idx + 1) % 2
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        routing_history.add_marker(f"Pal:{'Vir' if heatmap_palette_idx==1 else 'Tur'}", (26, 188, 156))
                    
        # ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ RENDERING ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬
        selected_route_name = active_selected_route(selected_route_name, routes)
        selected_room_poly = get_route_room_polygon(selected_route_name) if selected_route_name else None
        global_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
        selected_pins = get_selected_pin_names(selected_route_name, routes, global_pins) if selected_route_name else set()

        if show_heatmap:
            ensure_placement_heatmap_scores()

        def draw_edge_weight_overlay():
            if edge_weight_heatmap_enabled:
                refresh_edge_weight_view_overlay(routes)
            draw_edge_weight_heatmap(screen)

        canvas_scene = build_canvas_scene(CanvasFrameState(
            rooms=rooms, doors=doors, columns=columns, shafts=shafts,
            shaft_extraction=shaft_extraction, graph_env=routing_workspace.env,
            show_grid_graph=show_grid_graph, terminals=terminals, routes=routes or (),
            selected_route_name=selected_route_name, selected_room_polygon=selected_room_poly,
            global_pins=global_pins, selected_pins=selected_pins,
            auto_placement_mode=auto_placement_mode_idx, placement_fields=ap_fields,
            wet_room_names=wet_room_names, route_colors=ROUTE_COLORS,
            palette=CanvasFramePalette(
                COLOR_BG, COLOR_WALL, COLOR_DOOR, COLOR_COLUMN, COLOR_SHAFT, COLOR_SHAFT_INACTIVE,
                COLOR_GRAPH_EDGE, COLOR_GRAPH_NODE, COLOR_ROOM, COLOR_ROOM_COVERED,
                COLOR_DESELECTED_ROOM, COLOR_DESELECTED_PIN, COLOR_DESELECTED_ROUTE,
                COLOR_PLAN_LABEL, COLOR_TEXT, COLOR_SELECTION_HALO,
            ),
            callbacks=CanvasFrameCallbacks(
                to_screen, get_representative_point, get_route_draw_width,
                routing_workspace.spatial_index.query,
            ),
            room_wall_width=WALL_DRAW_WIDTH,
            terminal_area_start=terminal_area_start_mm,
            terminal_area_end=terminal_area_end_mm if terminal_area_dragging else None,
        ))
        canvas_hooks = CanvasRenderHooks(
            draw_covers=lambda: draw_geometry_overlay(screen, covers, COLOR_COVER_OVERLAY),
            draw_distance_heatmap=lambda: draw_distance_heatmap(screen, ap_scores) if show_heatmap and ap_scores else None,
            draw_distance_colorbar=lambda: draw_colorbar(screen, get_placeable_heatmap_scores(ap_scores)) if show_heatmap and ap_scores and not edge_weight_heatmap_enabled else None,
            draw_edge_weight_heatmap=draw_edge_weight_overlay,
            draw_edge_weight_colorbar=lambda: draw_edge_weight_colorbar(screen),
            draw_terminal_validity_overlay=lambda: draw_terminal_validity_overlay(screen),
            draw_wet_room_accents=lambda: draw_wet_room_outer_accents(screen),
            draw_polygon_hatch=lambda shaft: draw_polygon_hatch(screen, shaft, COLOR_SHAFT_INACTIVE_HATCH, spacing=9),
            draw_outlined_text=lambda draw_screen, font, label, position, color: draw_outlined_text(draw_screen, font, label, position, color),
            draw_preferred_terminal_areas=lambda selected: draw_preferred_terminal_areas(screen, selected),
            draw_routed_terminal_endpoints=lambda selected: draw_routed_terminal_endpoint_markers(screen, routes, selected),
            draw_preferred_terminal_markers=lambda selected: draw_preferred_terminal_markers(screen, selected, routes),
            draw_terminal_area_drag=lambda start, end: draw_terminal_area_drag(screen, start, end),
        )
        _draw_canvas_scene(screen, scene=canvas_scene, fonts=CanvasFonts(small=font_small), hooks=canvas_hooks)

        _draw_installation_header(
            screen,
            font_small,
            font_micro,
            installation_header_state,
            INSTALLATION_OPTIONS,
            get_installation_header_layout(),
            dragged_key=installation_drag_key,
        )

        draw_ruler_overlay(screen, font_small, ruler_start_mm, ruler_end_mm)
        draw_canvas_tool_controls(screen, font_small, ruler_mode, dwelling_selector_open)
        draw_terminal_tool_buttons(screen, font_bold, font_small)

        preferred_count = sum(
            len(points)
            for points in (() if terminal_runtime is None else terminal_runtime.preferred_points_by_room.values())
        )
        validation_warnings = get_route_validation_warnings(routes)
        sidebar_view = build_sidebar_view(SidebarFrameState(
            auto_placement_mode=AUTO_PLACEMENT_MODES[auto_placement_mode_idx],
            show_heatmap=show_heatmap, heatmap_scale_mode=heatmap_scale_mode,
            heatmap_palette_index=heatmap_palette_idx, weight_mode_index=weight_mode_idx,
            rotation_mode_index=rotation_mode_idx, rotation_field_scores=rotation_field_scores,
            machine_angle=machine_angle, strategy=ROUTING_STRATEGIES[routing_strategy_idx],
            router=ROUTER_BACKENDS[router_backend_idx], heuristic=HEURISTIC_MODES[heuristic_mode_idx],
            graph_type=GRAPH_TYPES[graph_type_idx], room_start_mode=ROOM_START_MODES[room_start_mode_idx],
            selected_route_name=selected_route_name, preferred_terminal_count=preferred_count,
            bend_value=C_BEND, bend_min=C_BEND_MIN, bend_max=C_BEND_MAX,
            crossing_value=crossing_penalty_multiplier, crossing_min=CROSSING_MULTIPLIER_MIN,
            crossing_max=CROSSING_MULTIPLIER_MAX, scenario_summary=current_scenario_summary,
            fallback_frame_name=ROUTING_FRAME_OPTIONS[routing_frame_idx],
            machine_position=(machine_cx, machine_cy), status=status,
            validation_warnings=validation_warnings, elapsed_ms=elapsed_ms,
            total_nodes=total_nodes, fps=clock.get_fps(),
        ))
        _draw_sidebar(
            screen,
            canvas_left=viewport.canvas_left,
            window_height=viewport.window_height,
            fonts=SidebarFonts(title=font_title, bold=font_bold, small=font_small),
            colors=SidebarColors(
                panel=COLOR_PANEL,
                wall=COLOR_WALL,
                text=COLOR_TEXT,
                muted=COLOR_MUTED,
            ),
            view=sidebar_view,
            draw_help_button=lambda card_id, rect, font: draw_card_help_button(screen, card_id, rect, font),
            draw_min_piece_slider=draw_min_piece_slider,
            draw_weight_slider=draw_weight_slider,
        )
        draw_viewer_legend(screen, font_small)

        panel_x = viewport.window_width - PANEL_W
        pygame.draw.rect(screen, COLOR_PANEL, (panel_x, 0, PANEL_W, viewport.window_height))
        pygame.draw.line(screen, (55, 55, 70), (panel_x, 0), (panel_x, viewport.window_height))
        lbl_panel = font_bold.render("PLACEMENT EXPLORER", True, COLOR_MUTED)
        screen.blit(lbl_panel, (panel_x + PANEL_W // 2 - lbl_panel.get_width() // 2, 8))
        draw_plots(screen, font_plot_small, font_plot_title, font_plot_value, font_plot_minimum)
        draw_solution_logs_panel(screen, font_small, font_bold)
        draw_help_popup(screen, font_small)
        draw_transient_message(screen, font_small)
        draw_terminal_validity_tooltip(screen, font_small)

        pygame.display.flip()
        clock.tick(FPS)
        
    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        error_log = Path(__file__).with_name("runtime_stderr.log")
        with error_log.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        raise
