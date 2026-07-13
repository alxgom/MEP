import sys
import os
import math
import time
import itertools
import heapq
from pathlib import Path
from collections import deque
import numpy as np
import pygame
from shapely.geometry import Polygon, LineString, Point, box, MultiLineString
from shapely.ops import unary_union
from shapely.affinity import scale as shapely_scale
from shapely.prepared import prep as shapely_prep
from scipy.spatial import cKDTree
from mep_routing.domain import (
    local_axis_to_world as _local_axis_to_world,
    machine_pins as _machine_pins,
    outward_vector as _outward_vector,
    port_access_specs as _port_access_specs,
)
from mep_routing.installations.sal import (
    LARGE_DUCT_ROUTE_NAMES,
    SAL_OZEO_FLAT_MACHINE,
    run_sequential_routing as _run_sal_sequential_routing,
    search_large_route_candidates as _search_sal_large_route_candidates,
    select_two_stage_routing as _select_sal_two_stage_routing,
)
from mep_routing.geometry import (
    cast_rays_numpy as _cast_rays_numpy,
    edge_parallel_segment_min_distances as _edge_parallel_segment_min_distances,
    edge_segment_min_distances as _edge_segment_min_distances,
    extract_boundary_segments as _extract_bnd_segs,
    extract_line_segments as _extract_line_segs,
    iter_polygons as _iter_polygons_from_geom,
    largest_polygon as _largest_polygon_from_geom,
    point_segment_min_distances as _point_segment_min_distances,
    ray_ray_intersections_numpy as _ray_ray_intersections_numpy,
    snap_to_integer_grid,
)
from mep_routing.graphs import (
    EnvView,
    build_axis_grid as _build_axis_grid_for_context,
    build_hannan_static_axes as _build_hannan_static_axes_for_context,
    build_epsilon_axes as _build_epsilon_axes_for_context,
    filter_dynamic_machine_obstacle as _filter_dynamic_machine_obstacle,
    add_bounds_axes as _add_bounds_axes_to_sets,
    add_epsilon_axis_values as _add_epsilon_axis_values_to_sets,
    add_epsilon_geometry_axes as _add_epsilon_geometry_axes_to_sets,
    add_point_axes as _add_point_axes_to_sets,
    add_polygon_vertex_axes as _add_polygon_vertex_axes_to_sets,
    extend_allowed_boundary_axes as _extend_allowed_boundary_axes_for_graph,
    merge_close_values as _merge_close_values_for_axes,
    build_regular_grid as _build_regular_grid_for_context,
)
from mep_routing.placement import (
    candidate_machine_rooms as _candidate_machine_rooms_for_placement,
    candidate_room_points as _candidate_room_points_for_placement,
    choose_core_like_machine_placement as _choose_core_like_machine_placement,
    choose_topological_machine_placement as _choose_topological_machine_placement,
    core_like_machine_candidate_score as _core_like_machine_candidate_score_for_placement,
    is_machine_placement_valid as _is_machine_placement_valid_for_placement,
    placement_weights as _placement_weights,
    routing_frame_axes as _routing_frame_axes_for_placement,
    score_rotation_field_at as _score_rotation_field_at_for_placement,
    select_field_alignment_rotation as _select_field_alignment_rotation,
    topological_placement_scores as _topological_placement_scores,
)
from mep_routing.routing import (
    RouteScoreWeights,
    block_terminal_node_edges as _block_terminal_node_edges,
    build_routes_from_paths as _build_routes_from_paths_for_env,
    buffered_radius_mm as _buffered_radius_mm,
    count_ordered_route_turns as _count_ordered_route_turns,
    count_route_short_pieces as _count_route_short_pieces,
    count_segment_clearance_conflicts as _count_segment_clearance_conflicts,
    count_segment_crossings as _count_segment_crossings,
    count_segment_overlaps as _count_segment_overlaps,
    count_solution_short_pieces as _count_solution_short_pieces,
    count_solution_turns as _count_solution_turns,
    add_port_stub_segment as _add_port_stub_segment,
    append_allowed_region_warning as _append_allowed_region_warning,
    build_pin_min_cost_flow_network as _build_pin_min_cost_flow_network,
    find_route_at_point as _find_route_at_point,
    find_route_hit_at_point as _find_route_hit_at_point,
    line_graph_dir_from_points as _line_graph_dir_from_points_for_env,
    merged_route_piece_lengths as _merged_route_piece_lengths,
    ordered_small_room_names as _ordered_small_room_names,
    path_physical_length as _path_physical_length_for_env,
    route_conflict_summary as _route_conflict_summary,
    required_clearance_mm as _required_clearance_mm,
    route_axis_records as _route_axis_records_for_policy,
    route_quality_warnings as _route_quality_warnings,
    route_segments_from_path as _route_segments_from_path_for_env,
    score_routes as _score_routes,
    selected_pin_names as _selected_pin_names,
    set_block_weight as _set_block_weight,
    min_cost_flow as _min_cost_flow,
    positive_flow_edges as _positive_flow_edges,
    small_pin_target_specs as _small_pin_target_specs,
    source_start_nodes as _source_start_nodes_for_kd,
    terminal_node_indices as _terminal_node_indices_for_kd,
    terminal_validity_entries as _terminal_validity_entries,
    target_heuristic as _target_heuristic_for_env,
    total_route_length_m as _total_route_length_m,
    trace_flow_path as _trace_flow_path,
    weighted_edge_cost as _weighted_edge_cost_for_weights,
)
from mep_routing.ui import (
    cool_colormap as _cool_colormap_value,
    heatmap_color as _heatmap_color,
    score_to_heatmap_t as _score_to_heatmap_t_value,
    turbo_color as _turbo_color,
    viridis_color as _viridis_color,
)
from mep_routing.ui.drawing import (
    draw_dashed_polyline as _draw_dashed_polyline,
    draw_geometry_overlay as _draw_geometry_overlay,
    draw_outlined_text as _draw_outlined_text,
    draw_polygon_hatch as _draw_polygon_hatch,
)
from mep_routing.ui.controls import (
    canvas_tool_button_bounds as _canvas_tool_button_bounds,
    draw_min_piece_slider as _draw_min_piece_slider_widget,
    draw_weight_slider as _draw_weight_slider_widget,
    draw_weight_view_switch as _draw_weight_view_switch_widget,
    slider_value_from_x as _slider_value_from_x,
    weight_view_switch_bounds as _weight_view_switch_bounds,
)
from mep_routing.ui.solution_logs import (
    best_log_updates as _best_log_updates,
    draw_solution_logs_panel as _draw_solution_logs_panel_widget,
    metric_value_for_log as _metric_value_for_log_entry,
    solution_log_action as _solution_log_action,
    solution_kpis as _solution_kpis,
)

# Add relative paths to sys.path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '08-bend-aware-non-orthogonal')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

DWELLING_EXPORT_ROOT = Path(r"C:\Users\ALEXIS GOMEL\Documents\Dwelling-export")
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
WALL_THICKNESS = 150
GRID_SPACING   = 200    # mm — regular routing grid resolution
HANNAN_SCAFFOLD_SPACING = 600  # mm, static connectivity scaffold for dynamic Hannan axes
CORE_EPSILON_GRID_MM = 200
MACHINE_SPEC = SAL_OZEO_FLAT_MACHINE
SMALL_PIN_STUB_LENGTH = MACHINE_SPEC.small_pin_stub_length_mm
LARGE_PIN_STUB_LENGTH = MACHINE_SPEC.large_pin_stub_length_mm
MACHINE_CLEARANCE = 0
MACHINE_BODY_W = MACHINE_SPEC.body_width_mm
MACHINE_BODY_H = MACHINE_SPEC.body_height_mm
MACHINE_OVERALL_W = MACHINE_SPEC.overall_width_mm
MACHINE_SMALL_DUCT_D = MACHINE_SPEC.small_duct_diameter_mm
MACHINE_LARGE_DUCT_D = MACHINE_SPEC.large_duct_diameter_mm
DUCT_BUFFER_RATIO = 1.05
ROUTING_WALL_CLEARANCE_MM = 100
TERMINAL_REGULATION_CLEARANCE_MM = ROUTING_WALL_CLEARANCE_MM
BUFFER_ROOM_TERMINALES_AIRE_MM = 150
PATINEJO_CLEARANCE_MM = 200
SHAFT_ENTRY_SEARCH_MM = 700
SHAFT_ENTRY_MAX_CANDIDATES = 16
MACHINE_CLEARANCE_SOFT_MARGIN_MM = 150
FPS            = 20
WHEEL_ROTATE_COOLDOWN_MS = 180
C_BEND_DEFAULT = 4000.0
C_BEND_MIN = 0.0
C_BEND_MAX = 10000.0
CROSSING_MULTIPLIER_DEFAULT = 5.0
CROSSING_MULTIPLIER_MIN = 0.0
CROSSING_MULTIPLIER_MAX = 12.0
C_BEND         = C_BEND_DEFAULT  # Turn penalty in mm
crossing_penalty_multiplier = CROSSING_MULTIPLIER_DEFAULT
CROSSING_PENALTY = crossing_penalty_multiplier * C_BEND
CLEARANCE_PENALTY = CROSSING_PENALTY
OVERLAP_BLOCK_WEIGHT = 1e9
OVERLAP_SCORE_PENALTY = 50 * C_BEND
MIN_PIECE_FACTOR_DEFAULT = 1.05
MIN_PIECE_FACTOR_MIN = 0.50
MIN_PIECE_FACTOR_MAX = 2.00
SHORT_PIECE_SCORE_PENALTY = 2 * C_BEND
min_piece_factor = MIN_PIECE_FACTOR_DEFAULT

# Graph types
GRAPH_TYPES = [
    "Regular 200mm Grid",
    "Hannan Grid (numpy)",
    "Epsilon Grid (core-like numpy)",
]

ROUTING_STRATEGIES = [
    "Greedy (Dual-Sort)",
    "First Fit",
    "Best Fit",
    "Negotiated Congestion",
    "Negotiated Congestion (Favour Large)",
    "Min-Cost Flow (Small Pins)",
    "Min-Cost Flow (Two-Stage)"
]
routing_strategy_idx = 1

ROUTER_BACKENDS = [
    "State-expanded A*",
    "Line graph L(G) A*",
    "Line graph L(G) GBFS"
]
router_backend_idx = 0

HEURISTIC_MODES = [
    "Pin + bends",
    "Pin distance",
    "Machine envelope",
    "Zero",
]
heuristic_mode_idx = 0

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
show_heatmap = False
edge_weight_heatmap_enabled = False
edge_weight_view_mode_idx = 0
route_real_diameter_width_enabled = False
edge_weight_debug_map = {}
edge_weight_overlay_excluded_edges = set()
static_clearance_cache = {"key": None, "wall": None, "shaft": None}
geometry_distance_cache = {}
help_popup_card = None
transient_message = None
transient_message_until_ms = 0
help_button_rects = {}
preferred_terminal_tool_mode = None
preferred_terminal_points_by_room = {}
preferred_terminal_areas = []
room_start_node_cache = {}
terminal_tool_button_rects = {}
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

# Pygame Window Config
WINDOW_WIDTH, WINDOW_HEIGHT = 1700, 930
CANVAS_LEFT = 320
CANVAS_TOP = 40
PANEL_W = 280          # right-side plot panel
COLORBAR_W = 56        # reserved lane between drawing canvas and right-side plots
CANVAS_W = WINDOW_WIDTH - CANVAS_LEFT - PANEL_W - COLORBAR_W - 10
CANVAS_H = WINDOW_HEIGHT - CANVAS_TOP - 40
COLORBAR_LEFT = CANVAS_LEFT + CANVAS_W + 10
is_fullscreen = False
min_piece_slider_rect = pygame.Rect(0, 0, 1, 1)
bend_weight_slider_rect = pygame.Rect(0, 0, 1, 1)
crossing_weight_slider_rect = pygame.Rect(0, 0, 1, 1)
bend_weight_reset_rect = pygame.Rect(0, 0, 1, 1)
crossing_weight_reset_rect = pygame.Rect(0, 0, 1, 1)
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

REAL_DWELLING_DB = DWELLING_EXPORT_ROOT / "data" / "dwellings.sqlite"
PREFERRED_SHAFT_INSTALLATION = "Sal"
REAL_DWELLING_SCENARIOS = [
    ("0002_real_c90", "A_A1_P00_V01"),
    ("0004_real_7e4", "1_1_2_1"),
    ("0001_real_2b2", "01_01_P04_V06"),
    ("0001_real_2b2", "01_01_P01_V01"),
]
ROUTING_FRAME_OPTIONS = [
    "dominant_walls",
    "area_inertia_allowed",
    "area_inertia_rooms",
]
real_scenario_idx = 0
routing_frame_idx = 0
current_scenario_label = "synthetic"
current_scenario_summary = {}

BASE_SCALE_PX_PER_MM = min(CANVAS_W / 15000.0, CANVAS_H / 11000.0)
zoom_level = 1.0
view_pan_x_px = 0.0
view_pan_y_px = 0.0
SCALE_PX_PER_MM = BASE_SCALE_PX_PER_MM
OFFSET_X = CANVAS_LEFT + (CANVAS_W - 15000.0 * SCALE_PX_PER_MM) / 2
OFFSET_Y = CANVAS_TOP + (CANVAS_H - 11000.0 * SCALE_PX_PER_MM) / 2

# History buffers for the right-side plots
HIST_MAXLEN = 400
hist_length = deque(maxlen=HIST_MAXLEN)        # total duct length in metres
hist_score  = deque(maxlen=HIST_MAXLEN)         # weighted cost score
hist_turns  = deque(maxlen=HIST_MAXLEN)         # number of turns
hist_turns_per_len = deque(maxlen=HIST_MAXLEN)  # turns / meter of routed length
hist_exec_ms = deque(maxlen=HIST_MAXLEN)         # solve execution time in milliseconds
hist_sample_count = 0
hist_ap_idx = None                              # sample index of last auto-placement
hist_event_markers = []                         # list of (index, label, color) tuples
solution_logs = []
auto_best_logs = {}
selected_log_id = None
log_button_rect = pygame.Rect(0, 0, 1, 1)
log_row_rects = []
weight_mode_idx = 0                             # 0 for Default, 1 for Equal Weights
heatmap_scale_mode = 0                          # 0 for Linear (75% Saturation), 1 for Log Scale
heatmap_palette_idx = 0                         # 0 for Turbo, 1 for Viridis
heatmap_surface_cache = {"key": None, "surface": None}

def to_screen(x, y):
    sx = OFFSET_X + x * SCALE_PX_PER_MM
    sy = OFFSET_Y + (11000.0 - y) * SCALE_PX_PER_MM
    return int(sx), int(sy)

def to_mm(sx, sy):
    x = (sx - OFFSET_X) / SCALE_PX_PER_MM
    y = 11000.0 - (sy - OFFSET_Y) / SCALE_PX_PER_MM
    return x, y

def update_view_transform():
    global SCALE_PX_PER_MM, OFFSET_X, OFFSET_Y, heatmap_surface_cache
    SCALE_PX_PER_MM = BASE_SCALE_PX_PER_MM * zoom_level
    center_x = CANVAS_LEFT + CANVAS_W / 2.0
    center_y = CANVAS_TOP + CANVAS_H / 2.0
    OFFSET_X = center_x - 7500.0 * SCALE_PX_PER_MM + view_pan_x_px
    OFFSET_Y = center_y - 5500.0 * SCALE_PX_PER_MM + view_pan_y_px
    heatmap_surface_cache = {"key": None, "surface": None}

def update_window_layout(width, height):
    global WINDOW_WIDTH, WINDOW_HEIGHT, CANVAS_W, CANVAS_H, COLORBAR_LEFT, BASE_SCALE_PX_PER_MM
    WINDOW_WIDTH = max(1200, int(width))
    WINDOW_HEIGHT = max(720, int(height))
    CANVAS_W = max(320, WINDOW_WIDTH - CANVAS_LEFT - PANEL_W - COLORBAR_W - 10)
    CANVAS_H = max(320, WINDOW_HEIGHT - CANVAS_TOP - 40)
    COLORBAR_LEFT = CANVAS_LEFT + CANVAS_W + 10
    BASE_SCALE_PX_PER_MM = min(CANVAS_W / 15000.0, CANVAS_H / 11000.0)
    update_view_transform()

def set_zoom_level(new_zoom):
    global zoom_level
    zoom_level = max(0.5, min(6.0, float(new_zoom)))
    update_view_transform()

def zoom_at_screen_point(new_zoom, screen_pos):
    global view_pan_x_px, view_pan_y_px
    before = to_mm(screen_pos[0], screen_pos[1])
    set_zoom_level(new_zoom)
    after_sx, after_sy = to_screen(before[0], before[1])
    view_pan_x_px += screen_pos[0] - after_sx
    view_pan_y_px += screen_pos[1] - after_sy
    update_view_transform()

def reset_view_transform():
    global zoom_level, view_pan_x_px, view_pan_y_px
    zoom_level = 1.0
    view_pan_x_px = 0.0
    view_pan_y_px = 0.0
    update_view_transform()

def get_canvas_tool_buttons():
    return [
        (action, pygame.Rect(bounds), label)
        for action, bounds, label in _canvas_tool_button_bounds(CANVAS_LEFT, CANVAS_TOP)
    ]

def handle_canvas_tool_button_click(pos):
    global route_real_diameter_width_enabled
    for action, rect, _ in get_canvas_tool_buttons():
        if not rect.collidepoint(pos):
            continue
        if action == "in":
            set_zoom_level(zoom_level * 1.25)
        elif action == "out":
            set_zoom_level(zoom_level / 1.25)
        elif action == "reset":
            reset_view_transform()
        elif action == "diameter":
            route_real_diameter_width_enabled = not route_real_diameter_width_enabled
        return action
    if get_weight_view_switch_rect().collidepoint(pos):
        return "weight_view"
    return None

def get_weight_view_switch_rect(weights_rect=None):
    if weights_rect is None:
        weights_rect = next(rect for action, rect, _ in get_canvas_tool_buttons() if action == "weights")
    return pygame.Rect(_weight_view_switch_bounds((weights_rect.x, weights_rect.y, weights_rect.width, weights_rect.height)))

def min_piece_factor_from_slider_x(x):
    if min_piece_slider_rect.width <= 0:
        return min_piece_factor
    return _slider_value_from_x(x, min_piece_slider_rect, MIN_PIECE_FACTOR_MIN, MIN_PIECE_FACTOR_MAX)

def set_min_piece_factor_from_slider_x(x):
    global min_piece_factor
    min_piece_factor = min_piece_factor_from_slider_x(x)

def slider_value_from_x(x, rect, min_value, max_value):
    return _slider_value_from_x(x, rect, min_value, max_value)

def refresh_route_weight_constants():
    global CROSSING_PENALTY, CLEARANCE_PENALTY, OVERLAP_SCORE_PENALTY, SHORT_PIECE_SCORE_PENALTY
    CROSSING_PENALTY = crossing_penalty_multiplier * C_BEND
    CLEARANCE_PENALTY = CROSSING_PENALTY
    OVERLAP_SCORE_PENALTY = 50 * C_BEND
    SHORT_PIECE_SCORE_PENALTY = 2 * C_BEND

def set_bend_weight_from_slider_x(x):
    global C_BEND
    raw_value = slider_value_from_x(x, bend_weight_slider_rect, C_BEND_MIN, C_BEND_MAX)
    C_BEND = float(round(raw_value / 100.0) * 100)
    refresh_route_weight_constants()

def set_crossing_weight_from_slider_x(x):
    global crossing_penalty_multiplier
    crossing_penalty_multiplier = slider_value_from_x(
        x,
        crossing_weight_slider_rect,
        CROSSING_MULTIPLIER_MIN,
        CROSSING_MULTIPLIER_MAX,
    )
    refresh_route_weight_constants()

def reset_bend_weight():
    global C_BEND
    C_BEND = C_BEND_DEFAULT
    refresh_route_weight_constants()

def reset_crossing_weight():
    global crossing_penalty_multiplier
    crossing_penalty_multiplier = CROSSING_MULTIPLIER_DEFAULT
    refresh_route_weight_constants()

def draw_min_piece_slider(screen, font_small, x, y, width):
    global min_piece_slider_rect
    min_piece_slider_rect = _draw_min_piece_slider_widget(
        screen, font_small, x, y, width, min_piece_factor,
        MIN_PIECE_FACTOR_MIN, MIN_PIECE_FACTOR_MAX, COLOR_TEXT, COLOR_MUTED,
    )

def draw_weight_slider(screen, font_small, x, y, width, label, value, min_value, max_value, color, rect_name, suffix="", integer=False):
    global bend_weight_slider_rect, crossing_weight_slider_rect, bend_weight_reset_rect, crossing_weight_reset_rect
    rect, reset_rect = _draw_weight_slider_widget(
        screen, font_small, x, y, width, label, value, min_value, max_value,
        COLOR_TEXT, COLOR_MUTED, suffix=suffix, integer=integer,
    )
    if rect_name == "bend":
        bend_weight_slider_rect = rect
        bend_weight_reset_rect = reset_rect
    else:
        crossing_weight_slider_rect = rect
        crossing_weight_reset_rect = reset_rect

def record_current_solution(routes, elapsed_ms, marker_label=None, marker_color=(241, 196, 15)):
    if routes:
        crossings_c = count_segment_crossings(routes)
        record_history(routes, crossings_c, elapsed_ms)
        if marker_label:
            hist_event_markers.append((hist_sample_count - 1, marker_label, marker_color))

def get_terminal_tool_buttons():
    x = CANVAS_LEFT + CANVAS_W - 145
    y = CANVAS_TOP + 92
    return [
        ("point", pygame.Rect(x, y, 132, 70), "Terminal"),
        ("area", pygame.Rect(x, y + 82, 132, 70), "Term. area"),
        ("map", pygame.Rect(x, y + 164, 132, 52), "Term. map"),
        ("reset", pygame.Rect(x, y + 226, 132, 34), "Reset prefs"),
    ]

def handle_terminal_tool_button_click(pos):
    global preferred_terminal_tool_mode, terminal_validity_overlay_enabled
    for mode, rect, _ in get_terminal_tool_buttons():
        if rect.collidepoint(pos):
            if mode == "reset":
                preferred_terminal_tool_mode = None
                return "reset"
            if mode == "map":
                terminal_validity_overlay_enabled = not terminal_validity_overlay_enabled
                return "map"
            preferred_terminal_tool_mode = None if preferred_terminal_tool_mode == mode else mode
            return "mode"
    return None

def draw_weight_view_switch(screen, font_small):
    rect = get_weight_view_switch_rect()
    _draw_weight_view_switch_widget(screen, font_small, rect, edge_weight_view_mode_idx == 0, COLOR_TEXT)

def draw_terminal_tool_buttons(screen, font_bold, font_small):
    global terminal_tool_button_rects
    terminal_tool_button_rects = {}
    for mode, rect, label in get_terminal_tool_buttons():
        terminal_tool_button_rects[mode] = rect
        active = terminal_validity_overlay_enabled if mode == "map" else preferred_terminal_tool_mode == mode
        fill = (45, 54, 80) if active else (38, 44, 54)
        border = (255, 255, 255) if active else (170, 180, 190)
        pygame.draw.rect(screen, fill, rect, border_radius=10)
        pygame.draw.rect(screen, border, rect, 3 if active else 2, border_radius=10)
        label_font = font_small if mode == "reset" else font_bold
        lbl = label_font.render(label, True, COLOR_TEXT)
        screen.blit(lbl, (rect.x + 12, rect.y + (8 if mode == "reset" else 10)))
        if mode == "reset":
            continue
        if mode == "point":
            icon = pygame.Rect(rect.right - 31, rect.y + 38, 16, 16)
            pygame.draw.rect(screen, COLOR_MUTED, icon, 2)
        elif mode == "area":
            icon = pygame.Rect(rect.right - 74, rect.y + 38, 58, 22)
            pygame.draw.rect(screen, (155, 89, 182), icon)
            pygame.draw.rect(screen, (255, 255, 255), icon, 2)
            for x in range(icon.left, icon.right, 8):
                pygame.draw.line(screen, (155, 89, 182), (x, icon.top), (x + 4, icon.top), 1)
                pygame.draw.line(screen, (155, 89, 182), (x, icon.bottom), (x + 4, icon.bottom), 1)
        else:
            icon = pygame.Rect(rect.right - 31, rect.y + 28, 16, 16)
            draw_dashed_polyline(
                screen,
                [icon.topleft, icon.topright, icon.bottomright, icon.bottomleft, icon.topleft],
                COLOR_TERMINAL_ALLOWED,
                2,
                dash_len=4,
                gap_len=3,
            )

def draw_canvas_tool_controls(screen, font_small, ruler_mode):
    for action, rect, label in get_canvas_tool_buttons():
        active = (
            (action == "ruler" and ruler_mode)
            or (action == "weights" and edge_weight_heatmap_enabled)
            or (action == "diameter" and route_real_diameter_width_enabled)
        )
        fill = (58, 80, 94) if active else (38, 44, 54)
        border = (52, 152, 219) if active else (170, 180, 190)
        pygame.draw.rect(screen, fill, rect, border_radius=4)
        pygame.draw.rect(screen, border, rect, 1, border_radius=4)
        lbl = font_small.render(label, True, COLOR_TEXT)
        screen.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.centery - lbl.get_height() // 2))
    draw_weight_view_switch(screen, font_small)
    zoom_lbl = font_small.render(f"{zoom_level:.2f}x", True, COLOR_TEXT)
    screen.blit(zoom_lbl, (CANVAS_LEFT + 12, CANVAS_TOP + 46))
    if edge_weight_heatmap_enabled:
        weight_view = "small pipe" if edge_weight_view_mode_idx == 0 else "big pipe"
        wgt_lbl = font_small.render(f"modified edge weights: {weight_view}", True, COLOR_TEXT)
        screen.blit(wgt_lbl, (CANVAS_LEFT + 86, CANVAS_TOP + 46))
    if preferred_terminal_tool_mode:
        hint = "click add, Ctrl+click erase" if preferred_terminal_tool_mode == "point" else "drag area, Ctrl+drag erase"
        term_lbl = font_small.render(f"terminal {preferred_terminal_tool_mode}: {hint}", True, COLOR_TEXT)
        screen.blit(term_lbl, (CANVAS_LEFT + 86, CANVAS_TOP + 64))

def set_ruler_cursor(enabled):
    try:
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR if enabled else pygame.SYSTEM_CURSOR_ARROW)
    except pygame.error:
        pass

def draw_ruler_overlay(screen, font_small, start_mm, end_mm):
    if start_mm is None or end_mm is None:
        return
    start_px = to_screen(start_mm[0], start_mm[1])
    end_px = to_screen(end_mm[0], end_mm[1])
    length_mm = math.hypot(end_mm[0] - start_mm[0], end_mm[1] - start_mm[1])

    pygame.draw.line(screen, (255, 255, 255), start_px, end_px, 5)
    pygame.draw.line(screen, (52, 152, 219), start_px, end_px, 3)
    pygame.draw.circle(screen, (255, 255, 255), start_px, 6)
    pygame.draw.circle(screen, (52, 152, 219), start_px, 4)
    pygame.draw.circle(screen, (255, 255, 255), end_px, 6)
    pygame.draw.circle(screen, (52, 152, 219), end_px, 4)

    label = font_small.render(f"{length_mm:.0f} mm", True, COLOR_TEXT)
    mid_x = (start_px[0] + end_px[0]) // 2
    mid_y = (start_px[1] + end_px[1]) // 2
    pad = 4
    label_rect = pygame.Rect(mid_x + 10, mid_y - label.get_height() - 8, label.get_width() + 2 * pad, label.get_height() + 2 * pad)
    pygame.draw.rect(screen, (22, 22, 30), label_rect, border_radius=4)
    pygame.draw.rect(screen, (52, 152, 219), label_rect, 1, border_radius=4)
    screen.blit(label, (label_rect.x + pad, label_rect.y + pad))

# Global layout variables
rooms = []
wet_room_outer_accents = []
columns = []
shafts = []
covers = []
doors = []
walls = []
wall_polys = []
routing_region_base = None
shaft_extraction = None
shaft_core_entry_specs = []
shaft_entry_geometry_by_node = {}
terminals        = {}
wet_room_names   = []
machine_cx    = 0.0
machine_cy    = 0.0
machine_angle = 0

# Grid cache
graph_type_idx  = 1
grid_nodes      = None
grid_adj_base   = None
grid_edge_list  = None
grid_edge_coords= None
grid_kd         = None
current_env     = None
_bnd_segs       = None
hannan_static_cache = {}
show_grid_graph = False

def invalidate_room_start_node_cache():
    global room_start_node_cache
    room_start_node_cache = {}

def invalidate_terminal_validity_cache():
    global terminal_validity_cache
    terminal_validity_cache = {"key": None, "entries": [], "reasons_by_node": {}}

# Auto-placement cache
ap_scores = {}
ap_fields = {}
base_regular_env = None
base_regular_kd = None

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
    env = env or current_env
    if env is None or node_idx is None:
        return None

    p = env.nodes[int(node_idx)]
    node_pt = Point(float(p[0]), float(p[1]))
    rep_x, rep_y = get_representative_point(shaft_extraction)
    rep = np.array([float(rep_x), float(rep_y)], dtype=np.float64)

    if shaft_extraction.contains(node_pt):
        return {
            "rep": (float(rep[0]), float(rep[1])),
            "entry": (float(p[0]), float(p[1])),
            "node": (float(p[0]), float(p[1])),
            "distance": 0.0,
            "orthogonality_error": 0.0,
        }

    boundary = shaft_extraction.boundary
    entry_pt = boundary.interpolate(boundary.project(node_pt))
    entry = np.array([float(entry_pt.x), float(entry_pt.y)], dtype=np.float64)
    node = np.array([float(p[0]), float(p[1])], dtype=np.float64)

    outward = node - entry
    radial = entry - rep
    outward_norm = float(np.linalg.norm(outward))
    radial_norm = float(np.linalg.norm(radial))
    if outward_norm < 1e-6 or radial_norm < 1e-6:
        orthogonality_error = 1.0
    else:
        if float(np.dot(outward, radial)) < 0.0:
            radial = -radial
        cos_align = abs(float(np.dot(outward, radial)) / (outward_norm * radial_norm))
        orthogonality_error = 1.0 - min(1.0, cos_align)

    return {
        "rep": (float(rep[0]), float(rep[1])),
        "entry": (float(entry[0]), float(entry[1])),
        "node": (float(node[0]), float(node[1])),
        "distance": outward_norm,
        "orthogonality_error": orthogonality_error,
    }

def get_shaft_entry_nodes(env, kd=None):
    global shaft_entry_geometry_by_node
    if shaft_extraction is None or env is None or len(env.nodes) == 0:
        return [], None

    shaft_entry_geometry_by_node = {}
    if shaft_core_entry_specs:
        candidates = []
        for spec_idx, spec in enumerate(shaft_core_entry_specs):
            entry = np.array(spec["entry"], dtype=np.float64)
            centroid = np.array(spec["centroid"], dtype=np.float64)
            normal = np.array(spec["normal"], dtype=np.float64)
            exit_wall = spec.get("exit_wall")
            search_indices = range(len(env.nodes))
            if kd is not None:
                found = kd.query_ball_point(entry, SHAFT_ENTRY_SEARCH_MM)
                if found:
                    search_indices = found
            for idx in search_indices:
                p = env.nodes[int(idx)]
                node = np.array([float(p[0]), float(p[1])], dtype=np.float64)
                node_pt = Point(float(node[0]), float(node[1]))
                if shaft_extraction.contains(node_pt):
                    continue
                offset = node - entry
                dist = float(np.linalg.norm(offset))
                if dist > SHAFT_ENTRY_SEARCH_MM:
                    continue
                offset_norm = float(np.linalg.norm(offset))
                if offset_norm < 1e-6:
                    align = 1.0
                else:
                    align = float(np.dot(offset / offset_norm, normal))
                if align < -1e-6:
                    continue
                orthogonality_error = 1.0 - max(0.0, min(1.0, align))
                exit_wall_penalty = 0.0
                if exit_wall is not None:
                    wall_line = LineString(exit_wall)
                    exit_wall_penalty = min(CORE_EPSILON_GRID_MM, wall_line.distance(Point(float(entry[0]), float(entry[1]))))
                shaft_distance = node_pt.distance(shaft_extraction)
                score = dist + orthogonality_error * GRID_SPACING * 4.0 + shaft_distance * 0.25 + exit_wall_penalty
                geom = {
                    "rep": (float(centroid[0]), float(centroid[1])),
                    "entry": (float(entry[0]), float(entry[1])),
                    "node": (float(node[0]), float(node[1])),
                    "distance": dist,
                    "orthogonality_error": orthogonality_error,
                    "source": "routing_core",
                }
                old = shaft_entry_geometry_by_node.get(int(idx))
                if old is None or score < old.get("score", float("inf")):
                    geom["score"] = score
                    geom["spec_idx"] = spec_idx
                    shaft_entry_geometry_by_node[int(idx)] = geom
                candidates.append((score, dist, orthogonality_error, int(idx)))

        if candidates:
            candidates.sort()
            chosen = []
            seen = set()
            for _, _, _, idx in candidates:
                if idx in seen:
                    continue
                seen.add(idx)
                chosen.append(idx)
                if len(chosen) >= SHAFT_ENTRY_MAX_CANDIDATES:
                    break
            return chosen, chosen[0]

    candidates = []
    for idx, pt in enumerate(env.nodes):
        node_pt = Point(float(pt[0]), float(pt[1]))
        if shaft_extraction.contains(node_pt):
            continue
        dist = node_pt.distance(shaft_extraction)
        if dist > SHAFT_ENTRY_SEARCH_MM:
            continue
        geom = _shaft_entry_geometry_for_node(idx, env)
        if geom is None:
            continue
        score = dist + geom["orthogonality_error"] * GRID_SPACING * 2.0
        candidates.append((score, dist, geom["orthogonality_error"], int(idx)))

    if candidates:
        candidates.sort()
        return [idx for _, _, _, idx in candidates[:SHAFT_ENTRY_MAX_CANDIDATES]], candidates[0][3]

    rep_pt = shaft_extraction.representative_point()
    if kd is not None:
        _, fallback_idx = kd.query((round(rep_pt.x), round(rep_pt.y)))
        return [int(fallback_idx)], int(fallback_idx)

    diffs = np.hypot(env.nodes[:, 0] - rep_pt.x, env.nodes[:, 1] - rep_pt.y)
    fallback_idx = int(np.argmin(diffs))
    return [fallback_idx], fallback_idx

def add_shaft_entry_segments(segs, first_node_idx):
    geom = _shaft_entry_geometry_for_node(first_node_idx)
    if geom is None:
        return

    rep = geom["rep"]
    entry = geom["entry"]
    node = geom["node"]

    if math.hypot(entry[0] - rep[0], entry[1] - rep[1]) > 1.0:
        segs.append((rep, entry))
    if math.hypot(node[0] - entry[0], node[1] - entry[1]) > 1.0:
        segs.append((entry, node))

def _commit_grid(nodes_arr, valid_edges):
    global grid_nodes, grid_adj_base, grid_edge_list, grid_edge_coords, grid_kd, current_env, static_clearance_cache, geometry_distance_cache
    DIR_REV = {'E': 'W', 'N': 'S', 'W': 'E', 'S': 'N'}

    if shaft_extraction is not None and len(nodes_arr) > 0:
        rep_pt = shaft_extraction.representative_point()
        sc = (round(rep_pt.x), round(rep_pt.y))
        
        minx_s, miny_s, maxx_s, maxy_s = shaft_extraction.bounds
        cx_s = (minx_s + maxx_s) / 2
        cy_s = (miny_s + maxy_s) / 2

        offset_val = ROUTING_WALL_CLEARANCE_MM
        face_pts = [
            (maxx_s + offset_val, cy_s, 'W'),
            (minx_s - offset_val, cy_s, 'E'),
            (cx_s, maxy_s + offset_val, 'S'),
            (cx_s, miny_s - offset_val, 'N')
        ]
        
        shaft_idx = len(nodes_arr)
        nodes_arr = np.vstack([nodes_arr, [sc[0], sc[1]]])
        
        connected_any = False
        for px, py, d in face_pts:
            diffs = np.hypot(nodes_arr[:-1, 0] - px, nodes_arr[:-1, 1] - py)
            min_idx = np.argmin(diffs)
            if diffs[min_idx] < 10.0:
                w = float(np.hypot(sc[0] - nodes_arr[min_idx, 0], sc[1] - nodes_arr[min_idx, 1]))
                valid_edges.append((min_idx, shaft_idx, w, d))
                connected_any = True
                
        if not connected_any:
            diffs = np.hypot(nodes_arr[:-1, 0] - sc[0], nodes_arr[:-1, 1] - sc[1])
            min_idx = np.argmin(diffs)
            dx = sc[0] - nodes_arr[min_idx, 0]
            dy = sc[1] - nodes_arr[min_idx, 1]
            if abs(dx) > abs(dy):
                d = 'E' if dx > 0 else 'W'
            else:
                d = 'N' if dy > 0 else 'S'
            w = float(np.hypot(dx, dy))
            valid_edges.append((min_idx, shaft_idx, w, d))

    pin_indices = {}
    global_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
    for name, pt in global_pins.items():
        diffs = np.hypot(nodes_arr[:, 0] - pt[0], nodes_arr[:, 1] - pt[1])
        min_idx = np.argmin(diffs)
        if diffs[min_idx] < 1.0:
            pin_indices[int(min_idx)] = name

    allowed_dirs_by_pin = {}
    for spec in get_port_access_specs(global_pins, machine_angle):
        allowed_dirs_by_pin.setdefault(spec["pin"], set()).add(spec["out_dir"])

    filtered_edges = []
    for u, v, w, d in valid_edges:
        keep = True
        if u in pin_indices:
            pin_name = pin_indices[u]
            allowed = allowed_dirs_by_pin.get(pin_name)
            if allowed and d not in allowed:
                keep = False
        if v in pin_indices:
            pin_name = pin_indices[v]
            allowed = allowed_dirs_by_pin.get(pin_name)
            if allowed and DIR_REV[d] not in allowed:
                keep = False
        if keep:
            filtered_edges.append((u, v, w, d))
            
    valid_edges = filtered_edges

    adj = {i: [] for i in range(len(nodes_arr))}
    for u, v, w, d in valid_edges:
        adj[u].append((v, w, d))
        adj[v].append((u, w, DIR_REV[d]))

    nodes_f32 = nodes_arr.astype(np.float32)
    if valid_edges:
        ec = np.array([[nodes_f32[u,0], nodes_f32[u,1],
                        nodes_f32[v,0], nodes_f32[v,1]]
                       for u, v, w, d in valid_edges], dtype=np.float32)
    else:
        ec = np.empty((0,4), dtype=np.float32)

    grid_nodes       = nodes_f32
    grid_adj_base    = adj
    grid_edge_list   = valid_edges
    grid_edge_coords = ec
    grid_kd          = cKDTree(grid_nodes)
    current_env      = EnvView(grid_nodes, adj)
    static_clearance_cache = {"key": None, "wall": None, "shaft": None}
    geometry_distance_cache = {}
    invalidate_room_start_node_cache()
    invalidate_terminal_validity_cache()

def _node_routing_region():
    if routing_region_base is None:
        return None
    if ROUTING_WALL_CLEARANCE_MM <= 0:
        return routing_region_base
    inset = routing_region_base.buffer(-ROUTING_WALL_CLEARANCE_MM, join_style=2)
    return routing_region_base if inset.is_empty else inset

def build_regular_grid():
    if routing_region_base is None:
        return
    nodes_arr, valid_edges = _build_regular_grid_for_context(
        routing_region_base, _node_routing_region(), wall_polys, GRID_SPACING, WALL_THICKNESS,
    )
    _commit_grid(nodes_arr, valid_edges)

def build_base_regular_grid():
    global base_regular_env, base_regular_kd
    if routing_region_base is None:
        return
    t0 = time.perf_counter()
    nodes_arr, valid_edges = _build_regular_grid_for_context(
        routing_region_base, _node_routing_region(), wall_polys, GRID_SPACING, WALL_THICKNESS,
    )
    
    DIR_REV = {'E': 'W', 'N': 'S', 'W': 'E', 'S': 'N'}
    adj = {i: [] for i in range(len(nodes_arr))}
    for u, v, w, d in valid_edges:
        adj[u].append((v, w, d))
        adj[v].append((u, w, DIR_REV[d]))
        
    base_regular_env = EnvView(nodes_arr.astype(np.float32), adj)
    base_regular_kd = cKDTree(base_regular_env.nodes)
    print(f"[Base Regular Grid] Built {len(nodes_arr)} nodes in {(time.perf_counter() - t0)*1000:.1f}ms")

def update_dynamic_env(machine_poly):
    global current_env
    if grid_nodes is None:
        current_env = None
        invalidate_room_start_node_cache()
        return

    t0 = time.perf_counter()
    protected_nodes = set()
    protected_points = list(terminals.values())
    protected_points.extend(
        spec["access_point"]
        for spec in get_port_access_specs(get_machine_pins(machine_cx, machine_cy, machine_angle), machine_angle)
    )
    for pt in protected_points:
        if str(pt).startswith("c_"):
            continue
        _, idx = grid_kd.query(pt)
        if np.hypot(grid_nodes[int(idx), 0] - pt[0], grid_nodes[int(idx), 1] - pt[1]) < 1.0:
            protected_nodes.add(int(idx))

    current_env, blocked_node_count, blocked_edge_count = _filter_dynamic_machine_obstacle(
        grid_nodes, grid_edge_list, grid_edge_coords, machine_poly, MACHINE_CLEARANCE, protected_nodes,
    )
    invalidate_room_start_node_cache()
    ms = (time.perf_counter() - t0) * 1000.0
    print(f"Grid update: {ms:.1f} ms  (blocked nodes={blocked_node_count}, edges={blocked_edge_count})")

def _iter_polygons(geom):
    yield from _iter_polygons_from_geom(geom)

def _add_point_axes(xs, ys, point):
    return _add_point_axes_to_sets(xs, ys, point)

def _add_polygon_vertex_axes(xs, ys, geom):
    return _add_polygon_vertex_axes_to_sets(xs, ys, geom)

def _add_bounds_axes(xs, ys, geom, clearance=0.0):
    return _add_bounds_axes_to_sets(xs, ys, geom, clearance)

def _largest_polygon(geom):
    return _largest_polygon_from_geom(geom)

def _extend_allowed_boundary_axes(allowed, inset=100.0, cluster_dist=300.0):
    return _extend_allowed_boundary_axes_for_graph(allowed, inset, cluster_dist)

def _merge_close_values(values, threshold, preserve_values=None, priority_values=None):
    return _merge_close_values_for_axes(values, threshold, preserve_values, priority_values)

def _get_hannan_static_template(shift_walls=False):
    global hannan_static_cache
    cache_key = bool(shift_walls)
    if cache_key in hannan_static_cache:
        return hannan_static_cache[cache_key]
    template = _build_hannan_static_axes_for_context(
        allowed_region=routing_region_base,
        terminals=terminals,
        shaft_extraction=shaft_extraction,
        covers=covers,
        columns=columns,
        shafts=shafts,
        wall_polys=wall_polys,
        walls=walls,
        grid_spacing_mm=GRID_SPACING,
        scaffold_spacing_mm=HANNAN_SCAFFOLD_SPACING,
        wall_clearance_mm=ROUTING_WALL_CLEARANCE_MM,
        shift_walls=shift_walls,
    )
    hannan_static_cache[cache_key] = template
    return template

def build_hannan_grid(machine_pins=None, shift_walls=False):
    if routing_region_base is None:
        return
    t0 = time.perf_counter()

    template = _get_hannan_static_template(shift_walls=shift_walls)
    xs = set(template["xs"])
    ys = set(template["ys"])
    preserve_x = set(template["preserve_x"])
    preserve_y = set(template["preserve_y"])
    required_points = []

    for pt in terminals.values():
        required_points.append((round(float(pt[0])), round(float(pt[1]))))
    if shaft_extraction is not None:
        rep_pt = shaft_extraction.representative_point()
        required_points.append((round(float(rep_pt.x)), round(float(rep_pt.y))))

    if machine_pins:
        for spec in get_port_access_specs(machine_pins, machine_angle):
            x, y = spec["access_point"]
            xs.add(x)
            ys.add(y)
            preserve_x.add(x)
            preserve_y.add(y)
            required_points.append((x, y))

    xs = _merge_close_values(xs, threshold=120.0, preserve_values=preserve_x, priority_values=template["priority_x"])
    ys = _merge_close_values(ys, threshold=120.0, preserve_values=preserve_y, priority_values=template["priority_y"])
    t1 = time.perf_counter()

    nodes_arr, raw_edges, (node_build_ms, edge_build_ms) = _build_axis_grid_for_context(
        xs, ys, routing_region_base, _node_routing_region(), wall_polys, WALL_THICKNESS, required_points,
    )
    if not len(nodes_arr):
        _commit_grid(nodes_arr, [])
        return

    _commit_grid(nodes_arr, raw_edges)
    ms_total = (time.perf_counter() - t0) * 1000.0
    ms_axes = (t1 - t0) * 1000.0
    ms_nodes = node_build_ms
    ms_edges = edge_build_ms
    print(
        f"[Hannan Simple] axes={len(xs)}x{len(ys)} nodes={len(nodes_arr)} edges={len(raw_edges)} "
        f"in {ms_total:.1f}ms (axes {ms_axes:.1f}, nodes {ms_nodes:.1f}, edges {ms_edges:.1f})"
    )

def _add_epsilon_axis_values(xs, ys, point, epsilon=CORE_EPSILON_GRID_MM):
    return _add_epsilon_axis_values_to_sets(xs, ys, point, epsilon)

def _add_epsilon_geometry_axes(xs, ys, geom, epsilon=CORE_EPSILON_GRID_MM):
    return _add_epsilon_geometry_axes_to_sets(xs, ys, geom, epsilon)

def build_epsilon_grid(machine_pins=None):
    if routing_region_base is None:
        return
    t0 = time.perf_counter()
    eps = CORE_EPSILON_GRID_MM
    machine_access_points = []
    if machine_pins:
        machine_access_points = [spec["access_point"] for spec in get_port_access_specs(machine_pins, machine_angle)]
    xs, ys, required_points = _build_epsilon_axes_for_context(
        allowed_region=routing_region_base,
        covers=covers,
        columns=columns,
        shafts=shafts,
        wall_polys=wall_polys,
        terminals=terminals,
        shaft_core_entry_specs=shaft_core_entry_specs,
        shaft_extraction=shaft_extraction,
        machine_access_points=machine_access_points,
        epsilon_mm=eps,
        scaffold_spacing_mm=HANNAN_SCAFFOLD_SPACING,
    )
    t1 = time.perf_counter()

    nodes_arr, raw_edges, (node_build_ms, edge_build_ms) = _build_axis_grid_for_context(
        xs, ys, routing_region_base, _node_routing_region(), wall_polys, WALL_THICKNESS, required_points,
    )
    if not len(nodes_arr):
        _commit_grid(nodes_arr, [])
        return

    _commit_grid(nodes_arr, raw_edges)
    ms_total = (time.perf_counter() - t0) * 1000.0
    print(
        f"[Epsilon Core-like] eps={eps:.0f} axes={len(xs)}x{len(ys)} nodes={len(nodes_arr)} "
        f"edges={len(raw_edges)} in {ms_total:.1f}ms "
        f"(axes {(t1-t0)*1000:.1f}, nodes {node_build_ms:.1f}, edges {edge_build_ms:.1f})"
    )

def build_grid(machine_pins=None):
    global grid_nodes, grid_adj_base, grid_edge_list, grid_edge_coords, grid_kd, current_env
    if graph_type_idx == 0:
        build_regular_grid()
    elif graph_type_idx == 1:
        build_hannan_grid(machine_pins=machine_pins, shift_walls=True)
    else:
        build_epsilon_grid(machine_pins=machine_pins)

# Machine representation helper
def get_machine_pins(cx, cy, angle_deg):
    return _machine_pins(MACHINE_SPEC, cx, cy, angle_deg)

def snap_pins_to_graph(global_pins):
    if grid_kd is None:
        return {}
    targets = {}
    for spec in get_port_access_specs(global_pins, machine_angle):
        _, idx = grid_kd.query(spec["access_point"])
        item = spec.copy()
        item["node_idx"] = int(idx)
        targets.setdefault(spec["pin"], []).append(item)
    return targets

# ──────────────────────────────────────────────────────────────────────────
# ROUTING UTILITIES AND CONSTRAINTS
# ──────────────────────────────────────────────────────────────────────────
def get_pin_stub_length(pin_name):
    return MACHINE_SPEC.pin_stub_length_mm(pin_name)

def get_port_access_specs(global_pins, machine_angle):
    return _port_access_specs(MACHINE_SPEC, global_pins, machine_angle)

def add_port_stub_segment(segs, pin_name, target_node_idx, global_pins, target_spec=None):
    return _add_port_stub_segment(segs, pin_name, target_node_idx, global_pins, current_env.nodes, target_spec)

def get_outward_vector(pin_name, machine_angle):
    return _outward_vector(pin_name, machine_angle)

def _route_axis_records(route_name, route_segs):
    return _route_axis_records_for_policy(route_name, route_segs, get_route_diameter)

def get_route_diameter(route_name):
    return MACHINE_SPEC.route_diameter_mm(route_name)

def get_buffered_radius_mm(diameter_mm):
    return _buffered_radius_mm(diameter_mm, DUCT_BUFFER_RATIO)

def get_required_clearance_mm(diameter_a, diameter_b):
    return _required_clearance_mm(diameter_a, diameter_b, DUCT_BUFFER_RATIO)

def _machine_edge_clearance_distances():
    if grid_edge_coords is None or len(grid_edge_coords) == 0:
        return None

    coords = grid_edge_coords.astype(np.float64, copy=False)
    samples_t = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    xs = coords[:, 0:1] + (coords[:, 2:3] - coords[:, 0:1]) * samples_t
    ys = coords[:, 1:2] + (coords[:, 3:4] - coords[:, 1:2]) * samples_t

    rad = math.radians(machine_angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    dx = xs - float(machine_cx)
    dy = ys - float(machine_cy)

    local_x = dx * cos_a + dy * sin_a
    local_y = -dx * sin_a + dy * cos_a
    half_w = MACHINE_OVERALL_W / 2.0
    half_h = MACHINE_BODY_H / 2.0

    outside_x = np.maximum(np.abs(local_x) - half_w, 0.0)
    outside_y = np.maximum(np.abs(local_y) - half_h, 0.0)
    return np.min(np.hypot(outside_x, outside_y), axis=1)

def _static_wall_distance_segments():
    segments = []
    if routing_region_base is not None and not routing_region_base.is_empty:
        bnd = _extract_bnd_segs(routing_region_base)
        if len(bnd):
            segments.append(bnd)
    for room in rooms:
        bnd = _extract_bnd_segs(room.polygon)
        if len(bnd):
            segments.append(bnd)
    for wall in walls:
        line = _extract_line_segs(wall)
        if len(line):
            segments.append(line)
    for wp in wall_polys:
        bnd = _extract_bnd_segs(wp)
        if len(bnd):
            segments.append(bnd)
    return np.vstack(segments) if segments else np.empty((0, 4), dtype=np.float64)

def _static_shaft_distance_segments():
    segments = []
    for shaft in shafts:
        bnd = _extract_bnd_segs(shaft)
        if len(bnd):
            segments.append(bnd)
    return np.vstack(segments) if segments else np.empty((0, 4), dtype=np.float64)

def _static_clearance_distances():
    global static_clearance_cache
    if grid_edge_coords is None or len(grid_edge_coords) == 0:
        return None, None

    key = (
        id(routing_region_base),
        len(grid_edge_list or []),
        len(rooms),
        len(wall_polys),
        len(shafts),
        tuple(room.polygon.bounds for room in rooms),
        tuple(poly.bounds for poly in shafts),
        tuple(poly.bounds for poly in wall_polys),
    )
    if static_clearance_cache.get("key") == key:
        return static_clearance_cache.get("wall"), static_clearance_cache.get("shaft")

    wall_distances = _edge_parallel_segment_min_distances(grid_edge_coords, _static_wall_distance_segments())
    shaft_distances = _edge_segment_min_distances(grid_edge_coords, _static_shaft_distance_segments())
    static_clearance_cache = {"key": key, "wall": wall_distances, "shaft": shaft_distances}
    return wall_distances, shaft_distances

def add_static_clearance_weights(edge_weights, route_diameter, env, allow_shaft_entry=False):
    wall_distances, shaft_distances = _static_clearance_distances()
    if wall_distances is None and shaft_distances is None:
        return

    radius = float(get_buffered_radius_mm(route_diameter))
    blocked_mask = np.zeros(len(grid_edge_list), dtype=bool)
    if wall_distances is not None:
        wall_limit = radius
        blocked_mask |= wall_distances < wall_limit - 1e-7
    if shaft_distances is not None and not allow_shaft_entry:
        shaft_limit = max(float(PATINEJO_CLEARANCE_MM), radius)
        blocked_mask |= shaft_distances < shaft_limit - 1e-7

    for ei in np.flatnonzero(blocked_mask):
        u, v, _, _ = grid_edge_list[int(ei)]
        edge_weights[(min(u, v), max(u, v))] = OVERLAP_BLOCK_WEIGHT

def add_machine_clearance_weights(edge_weights, route_diameter, env):
    if grid_edge_coords is None or grid_edge_list is None:
        return
    distances = _machine_edge_clearance_distances()
    if distances is None:
        return

    radius = float(get_buffered_radius_mm(route_diameter))
    soft_limit = radius + MACHINE_CLEARANCE_SOFT_MARGIN_MM
    hard_mask = distances < radius - 1e-7
    soft_mask = (distances >= radius - 1e-7) & (distances < soft_limit)

    for ei in np.flatnonzero(hard_mask):
        u, v, _, _ = grid_edge_list[int(ei)]
        edge_weights[(min(u, v), max(u, v))] = OVERLAP_BLOCK_WEIGHT

    soft_indices = np.flatnonzero(soft_mask)
    if len(soft_indices) == 0:
        return
    t = (soft_limit - distances[soft_indices]) / MACHINE_CLEARANCE_SOFT_MARGIN_MM
    penalties = CLEARANCE_PENALTY * np.square(t)
    for ei, penalty in zip(soft_indices, penalties):
        u, v, _, _ = grid_edge_list[int(ei)]
        edge = (min(u, v), max(u, v))
        if edge_weights.get(edge, 0.0) >= OVERLAP_BLOCK_WEIGHT:
            continue
        base_dist = float(np.hypot(env.nodes[v][0] - env.nodes[u][0], env.nodes[v][1] - env.nodes[u][1]))
        edge_weights[edge] = edge_weights.get(edge, base_dist) + float(penalty)

def add_route_clearance_weights(edge_weights, route_name, env):
    diameter = get_route_diameter(route_name)
    add_static_clearance_weights(
        edge_weights,
        diameter,
        env,
        allow_shaft_entry=(route_name == "Shaft"),
    )
    add_machine_clearance_weights(edge_weights, diameter, env)

def add_route_interaction_weights(prior_axis_records, current_diameter, accumulated_weights, env):
    if not prior_axis_records or grid_edge_coords is None or grid_edge_list is None:
        return

    coords = grid_edge_coords.astype(np.float64, copy=False)
    edge_x1 = np.minimum(coords[:, 0], coords[:, 2])
    edge_x2 = np.maximum(coords[:, 0], coords[:, 2])
    edge_y1 = np.minimum(coords[:, 1], coords[:, 3])
    edge_y2 = np.maximum(coords[:, 1], coords[:, 3])
    edge_is_h = np.abs(coords[:, 1] - coords[:, 3]) < 1e-7

    blocked_edges = set()
    crossing_counts = {}
    clearance_counts = {}

    for prior_seg, prior_diameter in prior_axis_records:
        px1, py1, px2, py2, prior_dir = prior_seg
        required = get_required_clearance_mm(current_diameter, prior_diameter)

        if prior_dir == "H":
            overlap_mask = (
                edge_is_h
                & (np.abs(edge_y1 - py1) < 1e-7)
                & (np.minimum(edge_x2, px2) - np.maximum(edge_x1, px1) > 1e-7)
            )
            cross_mask = (
                ~edge_is_h
                & (edge_x1 >= px1 - 1e-7)
                & (edge_x1 <= px2 + 1e-7)
                & (edge_y1 <= py1 + 1e-7)
                & (edge_y2 >= py1 - 1e-7)
            )
        else:
            overlap_mask = (
                ~edge_is_h
                & (np.abs(edge_x1 - px1) < 1e-7)
                & (np.minimum(edge_y2, py2) - np.maximum(edge_y1, py1) > 1e-7)
            )
            cross_mask = (
                edge_is_h
                & (edge_y1 >= py1 - 1e-7)
                & (edge_y1 <= py2 + 1e-7)
                & (edge_x1 <= px1 + 1e-7)
                & (edge_x2 >= px1 - 1e-7)
            )

        dx = np.maximum.reduce([px1 - edge_x2, edge_x1 - px2, np.zeros_like(edge_x1)])
        dy = np.maximum.reduce([py1 - edge_y2, edge_y1 - py2, np.zeros_like(edge_y1)])
        clearance_mask = (np.hypot(dx, dy) < required) & ~overlap_mask & ~cross_mask

        for ei in np.flatnonzero(overlap_mask):
            u, v, _, _ = grid_edge_list[int(ei)]
            blocked_edges.add((min(u, v), max(u, v)))
        for ei in np.flatnonzero(cross_mask):
            u, v, _, _ = grid_edge_list[int(ei)]
            edge = (min(u, v), max(u, v))
            crossing_counts[edge] = crossing_counts.get(edge, 0) + 1
        for ei in np.flatnonzero(clearance_mask):
            u, v, _, _ = grid_edge_list[int(ei)]
            edge = (min(u, v), max(u, v))
            clearance_counts[edge] = clearance_counts.get(edge, 0) + 1

    for edge in blocked_edges:
        accumulated_weights[edge] = OVERLAP_BLOCK_WEIGHT

    for edge in set(crossing_counts) | set(clearance_counts):
        if accumulated_weights.get(edge, 0.0) >= OVERLAP_BLOCK_WEIGHT:
            continue
        u, v = edge
        base_dist = float(np.hypot(env.nodes[v][0] - env.nodes[u][0], env.nodes[v][1] - env.nodes[u][1]))
        base_cost = accumulated_weights.get(edge, base_dist)
        accumulated_weights[edge] = (
            base_cost
            + CROSSING_PENALTY * crossing_counts.get(edge, 0)
            + CLEARANCE_PENALTY * clearance_counts.get(edge, 0)
        )

def _weighted_edge_cost(edge_weights, u, v, dist):
    return _weighted_edge_cost_for_weights(edge_weights, u, v, dist)

def set_terminal_block_weight(edge_weights, u, v):
    edge = _set_block_weight(edge_weights, u, v, OVERLAP_BLOCK_WEIGHT)
    edge_weight_overlay_excluded_edges.add(edge)

def record_edge_weight_overlay(edge_weights, env):
    if not edge_weights or env is None:
        return
    for (u, v), cost in edge_weights.items():
        if u < 0 or v < 0 or u >= len(env.nodes) or v >= len(env.nodes):
            continue
        base_len = float(np.hypot(env.nodes[v][0] - env.nodes[u][0], env.nodes[v][1] - env.nodes[u][1]))
        if base_len <= 1e-7:
            continue
        edge = (min(int(u), int(v)), max(int(u), int(v)))
        if edge in edge_weight_overlay_excluded_edges:
            continue
        if cost >= OVERLAP_BLOCK_WEIGHT:
            edge_weight_debug_map[edge] = max(edge_weight_debug_map.get(edge, 0.0), OVERLAP_BLOCK_WEIGHT)
            continue
        added = float(cost) - base_len
        if added <= 1e-7:
            continue
        ratio = added / base_len
        edge_weight_debug_map[edge] = max(edge_weight_debug_map.get(edge, 0.0), ratio)

def refresh_edge_weight_view_overlay(routes):
    global edge_weight_debug_map, edge_weight_overlay_excluded_edges
    edge_weight_debug_map = {}
    edge_weight_overlay_excluded_edges = set()
    if current_env is None:
        return

    diameter = MACHINE_SMALL_DUCT_D if edge_weight_view_mode_idx == 0 else MACHINE_LARGE_DUCT_D
    weights = {}
    add_static_clearance_weights(weights, diameter, current_env, allow_shaft_entry=False)
    add_machine_clearance_weights(weights, diameter, current_env)

    prior_axis_records = []
    if routes:
        for route_name, segs in routes:
            prior_axis_records.extend(_route_axis_records(route_name, segs))
    add_route_interaction_weights(prior_axis_records, diameter, weights, current_env)
    record_edge_weight_overlay(weights, current_env)

def _line_graph_dir_from_points(env, u, v):
    return _line_graph_dir_from_points_for_env(env, u, v)

def _path_physical_length(env, path):
    return _path_physical_length_for_env(env, path)

def _target_heuristic(env, node_idx, incoming_dir, target_specs, C_bend):
    return _target_heuristic_for_env(
        env,
        node_idx,
        incoming_dir,
        target_specs,
        C_bend,
        heuristic_mode_idx,
        (machine_cx, machine_cy),
        estimate_turns,
    )

def _run_super_sink_state_astar(env, start_node_indices, target_pin_names, pin_node_map, global_pins, machine_angle, C_bend, edge_weights=None):
    if isinstance(start_node_indices, (int, np.integer)):
        start_node_indices = [start_node_indices]
    if not target_pin_names or not start_node_indices:
        return None, 0.0, None, None
        
    num_nodes = len(env.nodes)
    super_source_idx = num_nodes
    super_sink_idx = num_nodes + 1
    
    search_nodes = np.zeros((num_nodes + 2, 2), dtype=np.float32)
    search_nodes[:num_nodes] = env.nodes
    search_nodes[super_source_idx] = env.nodes[start_node_indices[0]]
    search_nodes[super_sink_idx] = (machine_cx, machine_cy)
    
    search_adj = {i: list(env.adj[i]) for i in env.adj}
    search_adj[super_source_idx] = []
    search_adj[super_sink_idx] = []
    
    for start_node in start_node_indices:
        search_adj[super_source_idx].append((int(start_node), 0.0, None))
        
    target_specs = [
        target
        for pin_name in target_pin_names
        for target in pin_node_map.get(pin_name, [])
    ]
    if not target_specs:
        return None, 0.0, None, None

    pin_target_by_entry = {}
    for target in target_specs:
        pin_idx = int(target["node_idx"])
        pin_target_by_entry[(pin_idx, target["in_dir"])] = target
        search_adj[pin_idx].append((super_sink_idx, 0.0, target["in_dir"]))
        search_adj[super_sink_idx].append((pin_idx, 0.0, target["out_dir"]))
        
    search_env = EnvView(search_nodes, search_adj)
    pq = []
    counter = 0
    g_scores = {(super_source_idx, None): 0.0}
    came_from = {}
    visited = set()
    heapq.heappush(pq, (0.0, 0.0, counter, super_source_idx, None))
    best_target_state = None

    while pq:
        _, g, _, u, u_dir = heapq.heappop(pq)
        state = (u, u_dir)
        if state in visited:
            continue
        visited.add(state)
        if u == super_sink_idx:
            best_target_state = state
            break

        for v, dist, edge_dir in search_env.adj.get(u, []):
            v = int(v)
            edge_cost = _weighted_edge_cost(edge_weights, u, v, dist)
            turn_penalty = 0.0
            if u_dir is not None and edge_dir is not None and u_dir != edge_dir:
                turn_penalty = C_bend
            next_g = g + edge_cost + turn_penalty
            next_state = (v, edge_dir)
            if next_g < g_scores.get(next_state, float("inf")):
                g_scores[next_state] = next_g
                came_from[next_state] = state
                h = 0.0 if v >= num_nodes else _target_heuristic(env, v, edge_dir, target_specs, C_bend)
                counter += 1
                heapq.heappush(pq, (next_g + h, next_g, counter, v, edge_dir))
                
    if best_target_state is None:
        return None, 0.0, None, None

    states = []
    curr = best_target_state
    while curr in came_from:
        states.append(curr)
        curr = came_from[curr]
    states.append(curr)
    states.reverse()

    path = [state[0] for state in states]
    if len(path) < 3:
        return None, 0.0, None, None

    chosen_pin_idx = path[-2]
    chosen_target = pin_target_by_entry.get((chosen_pin_idx, best_target_state[1]))
    chosen_pin_name = chosen_target["pin"] if chosen_target else target_pin_names[0]
    path_without_virtual = path[1:-1]
    
    return path_without_virtual, _path_physical_length(env, path_without_virtual), chosen_pin_name, chosen_target

def _run_super_sink_line_graph_search(env, start_node_indices, target_pin_names, pin_node_map, global_pins, machine_angle, C_bend, edge_weights=None, greedy=False):
    if isinstance(start_node_indices, (int, np.integer)):
        start_node_indices = [start_node_indices]
    if not target_pin_names or not start_node_indices:
        return None, 0.0, None, None

    target_specs = [
        target
        for pin_name in target_pin_names
        for target in pin_node_map.get(pin_name, [])
    ]
    if not target_specs:
        return None, 0.0, None, None

    targets_by_node = {}
    for target in target_specs:
        node_idx = int(target["node_idx"])
        targets_by_node.setdefault(node_idx, []).append(target)

    pq = []
    counter = 0
    g_scores = {}
    came_from = {}
    state_dirs = {}

    for start_node in start_node_indices:
        for v, dist, edge_dir in env.adj.get(int(start_node), []):
            cost = _weighted_edge_cost(edge_weights, int(start_node), int(v), dist)
            state = (int(start_node), int(v))
            if cost < g_scores.get(state, float("inf")):
                g_scores[state] = cost
                state_dir = edge_dir if edge_dir is not None else _line_graph_dir_from_points(env, int(start_node), int(v))
                state_dirs[state] = state_dir
                h = _target_heuristic(env, int(v), state_dir, target_specs, C_bend)
                priority = h if greedy else cost + h
                heapq.heappush(pq, (priority, cost, counter, state))
                counter += 1

    best_final_cost = float("inf")
    best_final_state = None
    best_target = None
    visited = set()

    while pq:
        f_score, g, _, state = heapq.heappop(pq)
        if not greedy and f_score >= best_final_cost:
            break
        if state in visited:
            continue
        visited.add(state)

        u, v = state
        curr_dir = state_dirs[state]

        for target in targets_by_node.get(v, []):
            final_penalty = C_bend if curr_dir != target["in_dir"] else 0.0
            final_cost = g + final_penalty
            if final_cost < best_final_cost:
                best_final_cost = final_cost
                best_final_state = state
                best_target = target
        if greedy and best_final_state is not None:
            break

        for w, dist, next_dir in env.adj.get(v, []):
            w = int(w)
            if w == u:
                continue
            edge_cost = _weighted_edge_cost(edge_weights, v, w, dist)
            turn_penalty = C_bend if curr_dir != next_dir else 0.0
            next_state = (v, w)
            next_g = g + edge_cost + turn_penalty
            if next_g < g_scores.get(next_state, float("inf")):
                g_scores[next_state] = next_g
                came_from[next_state] = state
                state_dirs[next_state] = next_dir
                h = _target_heuristic(env, w, next_dir, target_specs, C_bend)
                priority = h if greedy else next_g + h
                heapq.heappush(pq, (priority, next_g, counter, next_state))
                counter += 1

    if best_final_state is None or best_target is None:
        return None, 0.0, None, None

    states = []
    curr = best_final_state
    while curr in came_from:
        states.append(curr)
        curr = came_from[curr]
    states.append(curr)
    states.reverse()

    path = [states[0][0]]
    path.extend(state[1] for state in states)
    return path, _path_physical_length(env, path), best_target["pin"], best_target

def _run_super_sink_line_graph_astar(env, start_node_indices, target_pin_names, pin_node_map, global_pins, machine_angle, C_bend, edge_weights=None):
    return _run_super_sink_line_graph_search(
        env, start_node_indices, target_pin_names, pin_node_map, global_pins, machine_angle, C_bend, edge_weights, greedy=False
    )

def _run_super_sink_line_graph_gbfs(env, start_node_indices, target_pin_names, pin_node_map, global_pins, machine_angle, C_bend, edge_weights=None):
    return _run_super_sink_line_graph_search(
        env, start_node_indices, target_pin_names, pin_node_map, global_pins, machine_angle, C_bend, edge_weights, greedy=True
    )

def run_super_sink_astar(env, start_node_indices, target_pin_names, pin_node_map, global_pins, machine_angle, C_bend, edge_weights=None):
    record_edge_weight_overlay(edge_weights, env)
    if router_backend_idx == 1:
        return _run_super_sink_line_graph_astar(
            env, start_node_indices, target_pin_names, pin_node_map, global_pins, machine_angle, C_bend, edge_weights
        )
    if router_backend_idx == 2:
        return _run_super_sink_line_graph_gbfs(
            env, start_node_indices, target_pin_names, pin_node_map, global_pins, machine_angle, C_bend, edge_weights
        )
    return _run_super_sink_state_astar(
        env, start_node_indices, target_pin_names, pin_node_map, global_pins, machine_angle, C_bend, edge_weights
    )

def get_all_terminal_node_indices(pin_node_map, shaft_node_idx):
    return _terminal_node_indices_for_kd(terminals, shaft_node_idx, grid_kd)

def _room_polygon_by_name(room_name):
    for room in rooms:
        if room.name == room_name:
            return room.polygon
    return None

def _room_cover_geometry(room_name):
    cache_key = ("room_cover_geometry", room_name, tuple(id(cover) for cover in covers), len(covers))
    if cache_key in geometry_distance_cache:
        return geometry_distance_cache[cache_key]

    room_poly = _room_polygon_by_name(room_name)
    if room_poly is None or room_poly.is_empty or not covers:
        geometry_distance_cache[cache_key] = None
        return None

    cover_parts = []
    for cover in covers:
        if cover is None or cover.is_empty or not cover.intersects(room_poly):
            continue
        part = cover.intersection(room_poly)
        if not part.is_empty and getattr(part, "area", 0.0) > 1e-6:
            cover_parts.append(part)

    if not cover_parts:
        geometry_distance_cache[cache_key] = None
        return None

    cover_geom = unary_union(cover_parts)
    geometry_distance_cache[cache_key] = cover_geom
    return cover_geom

def _room_terminal_valid_region(room_name):
    room_poly = _room_polygon_by_name(room_name)
    if room_poly is None or routing_region_base is None:
        return room_poly

    valid_region = room_poly.intersection(routing_region_base)
    cover_geom = _room_cover_geometry(room_name)
    if cover_geom is not None and not cover_geom.is_empty:
        valid_region = valid_region.intersection(cover_geom)
    return valid_region

def _room_terminal_boundary_segments(room_name):
    cache_key = (
        "room_terminal_boundary",
        room_name,
        id(routing_region_base),
        len(rooms),
        len(wall_polys),
        tuple(id(cover) for cover in covers),
    )
    if cache_key in geometry_distance_cache:
        return geometry_distance_cache[cache_key]

    room_poly = _room_polygon_by_name(room_name)
    cover_geom = _room_cover_geometry(room_name)
    segments = []
    if room_poly is not None:
        bnd = _extract_bnd_segs(room_poly)
        if len(bnd):
            segments.append(bnd)
    if cover_geom is not None and not cover_geom.is_empty:
        bnd = _extract_bnd_segs(cover_geom)
        if len(bnd):
            segments.append(bnd)
    for room in rooms:
        bnd = _extract_bnd_segs(room.polygon)
        if len(bnd):
            segments.append(bnd)
    for wall in walls:
        line = _extract_line_segs(wall)
        if len(line):
            segments.append(line)
    for wp in wall_polys:
        bnd = _extract_bnd_segs(wp)
        if len(bnd):
            segments.append(bnd)
    result = np.vstack(segments) if segments else np.empty((0, 4), dtype=np.float64)
    geometry_distance_cache[cache_key] = result
    return result

def _filter_terminal_candidate_nodes_by_wall_distance(room_name, candidate_nodes):
    if current_env is None or not candidate_nodes:
        return []
    segments = _room_terminal_boundary_segments(room_name)
    if len(segments) == 0:
        return list(candidate_nodes)

    node_indices = np.array(candidate_nodes, dtype=np.int64)
    pts = current_env.nodes[node_indices]
    distances = _point_segment_min_distances(pts, segments)
    min_clearance = max(TERMINAL_REGULATION_CLEARANCE_MM, BUFFER_ROOM_TERMINALES_AIRE_MM)
    keep = distances >= float(min_clearance) - 1e-7
    return [int(i) for i in node_indices[keep]]

def get_room_candidate_start_nodes(route_name):
    if grid_kd is None or current_env is None:
        return []
    if route_name in room_start_node_cache:
        return list(room_start_node_cache[route_name])

    terminal_pt = terminals.get(route_name)
    if terminal_pt is None:
        return []

    room_poly = _room_polygon_by_name(route_name)
    if room_poly is None or routing_region_base is None:
        _, node_idx = grid_kd.query(terminal_pt)
        nodes = [int(node_idx)]
        room_start_node_cache[route_name] = nodes
        return list(nodes)

    valid_region = _room_terminal_valid_region(route_name)
    if valid_region is None or valid_region.is_empty:
        room_start_node_cache[route_name] = []
        return []

    prepared = shapely_prep(valid_region)
    nodes = [
        int(i)
        for i, pt in enumerate(current_env.nodes)
        if current_env.adj.get(int(i)) and prepared.contains(Point(float(pt[0]), float(pt[1])))
    ]
    nodes = _filter_terminal_candidate_nodes_by_wall_distance(route_name, nodes)
    if nodes:
        nodes.sort(
            key=lambda i: float(
                np.hypot(current_env.nodes[i][0] - terminal_pt[0], current_env.nodes[i][1] - terminal_pt[1])
            )
        )
        room_start_node_cache[route_name] = nodes
        return list(nodes)

    nodes = []
    room_start_node_cache[route_name] = nodes
    return list(nodes)

def _preferred_points_for_room(route_name):
    return preferred_terminal_points_by_room.get(route_name, [])

def _map_preferred_points_to_nodes(route_name, candidate_nodes):
    prefs = _preferred_points_for_room(route_name)
    if not prefs or not candidate_nodes or current_env is None:
        return [], {}

    mapped_nodes = []
    mapped_pref_indices = {}
    candidate_arr = current_env.nodes[candidate_nodes]
    for pref_idx, pref_pt in enumerate(prefs):
        deltas = candidate_arr - np.array(pref_pt, dtype=np.float32)
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        nearest_pos = int(np.argmin(distances))
        if float(distances[nearest_pos]) > PREFERRED_TERMINAL_REMAP_TOLERANCE_MM:
            continue
        node_idx = int(candidate_nodes[nearest_pos])
        if node_idx not in mapped_pref_indices:
            mapped_nodes.append(node_idx)
            mapped_pref_indices[node_idx] = pref_idx
    return mapped_nodes, mapped_pref_indices

def get_route_start_nodes(route_name):
    if grid_kd is None or current_env is None:
        return []
    terminal_pt = terminals.get(route_name)
    if terminal_pt is None:
        return []
    if room_start_mode_idx == 1:
        _, node_idx = grid_kd.query(terminal_pt)
        return [int(node_idx)]

    candidate_nodes = get_room_candidate_start_nodes(route_name)
    if _preferred_points_for_room(route_name):
        preferred_nodes, _ = _map_preferred_points_to_nodes(route_name, candidate_nodes)
        return preferred_nodes
    return candidate_nodes

def _terminal_marker_side_px():
    return max(4, int(round(PREFERRED_TERMINAL_MARKER_SIZE_MM * SCALE_PX_PER_MM)))

def find_room_candidate_node_at_world(world_pt):
    if current_env is None:
        return None
    point = Point(float(world_pt[0]), float(world_pt[1]))
    best = None
    best_dist = float("inf")
    for room_name in terminals.keys():
        room_poly = _room_polygon_by_name(room_name)
        if room_poly is None or not (room_poly.contains(point) or room_poly.distance(point) < 1e-7):
            continue
        for node_idx in get_room_candidate_start_nodes(room_name):
            pt = current_env.nodes[int(node_idx)]
            dist = math.hypot(float(pt[0]) - world_pt[0], float(pt[1]) - world_pt[1])
            if dist < best_dist:
                best_dist = dist
                best = (room_name, int(node_idx))
    return best

def apply_preferred_terminal_point(world_pt, remove=False):
    global preferred_terminal_points_by_room
    hit = find_room_candidate_node_at_world(world_pt)
    if hit is None:
        return False, None

    room_name, node_idx = hit
    candidate_nodes = get_room_candidate_start_nodes(room_name)
    _, mapped_pref_indices = _map_preferred_points_to_nodes(room_name, candidate_nodes)
    prefs = list(preferred_terminal_points_by_room.get(room_name, []))

    if remove:
        pref_idx = mapped_pref_indices.get(node_idx)
        if pref_idx is None:
            return False, room_name
        del prefs[pref_idx]
        if prefs:
            preferred_terminal_points_by_room[room_name] = prefs
        else:
            preferred_terminal_points_by_room.pop(room_name, None)
        return True, room_name

    if node_idx in mapped_pref_indices:
        return False, room_name

    node_pt = current_env.nodes[node_idx]
    prefs.append((float(node_pt[0]), float(node_pt[1])))
    preferred_terminal_points_by_room[room_name] = prefs
    return True, room_name

def apply_preferred_terminal_area(start_world, end_world, remove=False):
    global preferred_terminal_points_by_room, preferred_terminal_areas
    if start_world is None or end_world is None or current_env is None:
        return False, None
    minx = min(float(start_world[0]), float(end_world[0]))
    maxx = max(float(start_world[0]), float(end_world[0]))
    miny = min(float(start_world[1]), float(end_world[1]))
    maxy = max(float(start_world[1]), float(end_world[1]))
    if maxx - minx < 1.0 or maxy - miny < 1.0:
        return False, None

    changed = False
    last_room = None
    for room_name in terminals.keys():
        candidate_nodes = get_room_candidate_start_nodes(room_name)
        if not candidate_nodes:
            continue
        prefs = list(preferred_terminal_points_by_room.get(room_name, []))
        if remove:
            kept = [
                pt for pt in prefs
                if not (minx <= pt[0] <= maxx and miny <= pt[1] <= maxy)
            ]
            if len(kept) != len(prefs):
                changed = True
                last_room = room_name
                if kept:
                    preferred_terminal_points_by_room[room_name] = kept
                else:
                    preferred_terminal_points_by_room.pop(room_name, None)
                preferred_terminal_areas = [
                    area for area in preferred_terminal_areas
                    if not (
                        area["room"] == room_name
                        and not (
                            area["bounds"][2] < minx or area["bounds"][0] > maxx
                            or area["bounds"][3] < miny or area["bounds"][1] > maxy
                        )
                    )
                ]
            continue

        _, mapped_pref_indices = _map_preferred_points_to_nodes(room_name, candidate_nodes)
        existing_nodes = set(mapped_pref_indices.keys())
        added_for_room = False
        for node_idx in candidate_nodes:
            if int(node_idx) in existing_nodes:
                continue
            pt = current_env.nodes[int(node_idx)]
            if minx <= float(pt[0]) <= maxx and miny <= float(pt[1]) <= maxy:
                prefs.append((float(pt[0]), float(pt[1])))
                existing_nodes.add(int(node_idx))
                changed = True
                added_for_room = True
                last_room = room_name
        if prefs:
            preferred_terminal_points_by_room[room_name] = prefs
        if added_for_room:
            preferred_terminal_areas.append({"room": room_name, "bounds": (minx, miny, maxx, maxy)})
    return changed, last_room

def draw_preferred_terminal_areas(screen, selected_route_name=None):
    if current_env is None:
        return
    marker_side = max(3, _terminal_marker_side_px() // 2)
    for area in preferred_terminal_areas:
        room_name = area["room"]
        if selected_route_name and room_name != selected_route_name:
            color = (86, 90, 94)
            node_color = (70, 74, 78)
        else:
            color = ROUTE_COLORS.get(room_name, (155, 89, 182))
            node_color = color
        minx, miny, maxx, maxy = area["bounds"]
        x1, y1 = to_screen(minx, miny)
        x2, y2 = to_screen(maxx, maxy)
        rect = pygame.Rect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        if rect.width > 1 and rect.height > 1:
            overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            overlay.fill((color[0], color[1], color[2], 45))
            screen.blit(overlay, rect.topleft)
            dash = 8
            for dx in range(0, rect.width, dash * 2):
                pygame.draw.line(screen, color, (rect.left + dx, rect.top), (min(rect.left + dx + dash, rect.right), rect.top), 2)
                pygame.draw.line(screen, color, (rect.left + dx, rect.bottom), (min(rect.left + dx + dash, rect.right), rect.bottom), 2)
            for dy in range(0, rect.height, dash * 2):
                pygame.draw.line(screen, color, (rect.left, rect.top + dy), (rect.left, min(rect.top + dy + dash, rect.bottom)), 2)
                pygame.draw.line(screen, color, (rect.right, rect.top + dy), (rect.right, min(rect.top + dy + dash, rect.bottom)), 2)
        for node_idx in get_room_candidate_start_nodes(room_name):
            pt = current_env.nodes[int(node_idx)]
            if minx <= float(pt[0]) <= maxx and miny <= float(pt[1]) <= maxy:
                sx, sy = to_screen(float(pt[0]), float(pt[1]))
                n_rect = pygame.Rect(0, 0, marker_side, marker_side)
                n_rect.center = (sx, sy)
                pygame.draw.rect(screen, node_color, n_rect, 1)

def draw_preferred_terminal_markers(screen, selected_route_name=None, routes=None):
    if current_env is None:
        return
    routed_start_by_room = {}
    if routes:
        for route_name, segs in routes:
            if segs:
                routed_start_by_room[route_name] = segs[0][0]
    marker_side = _terminal_marker_side_px()
    for room_name in terminals.keys():
        candidate_nodes = get_room_candidate_start_nodes(room_name)
        _, mapped_pref_indices = _map_preferred_points_to_nodes(room_name, candidate_nodes)
        route_color = ROUTE_COLORS.get(room_name, COLOR_TEXT)
        muted = bool(selected_route_name and room_name != selected_route_name)
        for node_idx in candidate_nodes:
            preferred = int(node_idx) in mapped_pref_indices
            if not preferred:
                continue
            pt = current_env.nodes[int(node_idx)]
            sx, sy = to_screen(float(pt[0]), float(pt[1]))
            rect = pygame.Rect(0, 0, marker_side, marker_side)
            rect.center = (sx, sy)
            routed_start = routed_start_by_room.get(room_name)
            is_routed = (
                routed_start is not None
                and abs(float(pt[0]) - routed_start[0]) < 1e-6
                and abs(float(pt[1]) - routed_start[1]) < 1e-6
            )
            if muted:
                fill = (44, 48, 52) if is_routed else None
                border = (86, 90, 94)
            else:
                fill = route_color if is_routed else None
                border = (255, 255, 255)
            if fill is not None:
                pygame.draw.rect(screen, fill, rect)
            pygame.draw.rect(screen, border, rect, 2)

def draw_routed_terminal_endpoint_markers(screen, routes, selected_route_name=None):
    if not routes:
        return
    marker_side = _terminal_marker_side_px()
    for route_name, segs in routes:
        if route_name not in terminals or not segs:
            continue
        start = segs[0][0]
        sx, sy = to_screen(float(start[0]), float(start[1]))
        rect = pygame.Rect(0, 0, marker_side, marker_side)
        rect.center = (sx, sy)
        if selected_route_name and route_name != selected_route_name:
            fill = (44, 48, 52)
            border = (86, 90, 94)
        else:
            fill = ROUTE_COLORS.get(route_name, COLOR_TEXT)
            border = (255, 255, 255)
        pygame.draw.rect(screen, fill, rect)
        pygame.draw.rect(screen, border, rect, 2)

def draw_geometry_overlay(screen, geometries, color_rgba):
    return _draw_geometry_overlay(screen, geometries, color_rgba, to_screen, (WINDOW_WIDTH, WINDOW_HEIGHT))

def draw_polygon_hatch(screen, poly, color, spacing=10):
    return _draw_polygon_hatch(screen, poly, color, to_screen, (WINDOW_WIDTH, WINDOW_HEIGHT), spacing)

def draw_dashed_polyline(screen, points, color, width=1, dash_len=8, gap_len=5):
    return _draw_dashed_polyline(screen, points, color, width, dash_len, gap_len)

def _terminal_validity_cache_key():
    return (
        id(current_env.nodes) if current_env is not None else None,
        len(current_env.nodes) if current_env is not None else 0,
        tuple(sorted(terminals.keys())),
        id(routing_region_base),
        len(rooms),
        len(walls),
        len(wall_polys),
        tuple(id(cover) for cover in covers),
    )

def get_terminal_validity_entries():
    if current_env is None:
        return [], {}

    key = _terminal_validity_cache_key()
    if terminal_validity_cache.get("key") == key:
        return terminal_validity_cache["entries"], terminal_validity_cache["reasons_by_node"]

    entries, reasons_by_node = _terminal_validity_entries(
        current_env.nodes,
        current_env.adj,
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
    x, y = center
    half = max(2, side // 2)
    rect = pygame.Rect(int(x - half), int(y - half), half * 2, half * 2)
    if allowed:
        draw_dashed_polyline(
            screen,
            [rect.topleft, rect.topright, rect.bottomright, rect.bottomleft, rect.topleft],
            COLOR_TERMINAL_ALLOWED,
            1,
            dash_len=4,
            gap_len=3,
        )
        return

    pygame.draw.rect(screen, COLOR_TERMINAL_BLOCKED, rect, 2)
    previous_clip = screen.get_clip()
    screen.set_clip(rect)
    for offset in range(-rect.height, rect.width + rect.height, 6):
        start = (rect.left + offset, rect.bottom)
        end = (rect.left + offset + rect.height, rect.top)
        pygame.draw.line(screen, COLOR_TERMINAL_BLOCKED_HATCH, start, end, 1)
    screen.set_clip(previous_clip)

def draw_terminal_validity_overlay(screen):
    if not terminal_validity_overlay_enabled:
        return
    entries, _ = get_terminal_validity_entries()
    marker_side = max(4, min(13, int(round(70 * SCALE_PX_PER_MM))))
    for x, y, _node_idx, allowed in entries:
        sx, sy = to_screen(x, y)
        if sx < CANVAS_LEFT - marker_side or sx > CANVAS_LEFT + CANVAS_W + marker_side:
            continue
        if sy < CANVAS_TOP - marker_side or sy > CANVAS_TOP + CANVAS_H + marker_side:
            continue
        draw_terminal_validity_square(screen, (sx, sy), marker_side, allowed)

def draw_terminal_validity_tooltip(screen, font_small):
    if not terminal_validity_overlay_enabled or grid_kd is None or current_env is None:
        return
    mx, my = pygame.mouse.get_pos()
    if not (CANVAS_LEFT <= mx <= CANVAS_LEFT + CANVAS_W and CANVAS_TOP <= my <= CANVAS_TOP + CANVAS_H):
        return

    world_pt = to_mm(mx, my)
    _, node_idx = grid_kd.query(world_pt)
    node_idx = int(node_idx)
    if node_idx < 0 or node_idx >= len(current_env.nodes):
        return

    node_pt = current_env.nodes[node_idx]
    sx, sy = to_screen(float(node_pt[0]), float(node_pt[1]))
    if math.hypot(mx - sx, my - sy) > 14:
        return

    _entries, reasons_by_node = get_terminal_validity_entries()
    reasons = reasons_by_node.get(node_idx, ["terminal status unknown"])
    lines = [f"node {node_idx}"] + reasons[:3]
    surfaces = [font_small.render(line, True, COLOR_TEXT) for line in lines]
    width = max(s.get_width() for s in surfaces) + 18
    height = len(surfaces) * 18 + 12
    rect = pygame.Rect(mx + 14, my + 14, width, height)
    if rect.right > WINDOW_WIDTH - 8:
        rect.right = mx - 14
    if rect.bottom > WINDOW_HEIGHT - 8:
        rect.bottom = my - 14
    pygame.draw.rect(screen, (32, 34, 38), rect, border_radius=5)
    pygame.draw.rect(screen, (130, 138, 146), rect, 1, border_radius=5)
    for i, surf in enumerate(surfaces):
        screen.blit(surf, (rect.x + 9, rect.y + 7 + i * 18))

def draw_wet_room_outer_accents(screen):
    for geom in wet_room_outer_accents:
        for poly in _iter_polygons(geom):
            coords = [to_screen(x, y) for x, y in poly.exterior.coords]
            pygame.draw.lines(screen, COLOR_WET_ROOM_ACCENT, True, coords, 3)

def draw_outlined_text(screen, font, text, pos, color, outline_color=COLOR_PLAN_LABEL_HALO):
    return _draw_outlined_text(screen, font, text, pos, color, outline_color)

def draw_terminal_area_drag(screen, start_world, end_world):
    if start_world is None or end_world is None:
        return
    x1, y1 = to_screen(start_world[0], start_world[1])
    x2, y2 = to_screen(end_world[0], end_world[1])
    rect = pygame.Rect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    if rect.width <= 1 or rect.height <= 1:
        return
    overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    overlay.fill((155, 89, 182, 55))
    screen.blit(overlay, rect.topleft)
    pygame.draw.rect(screen, (255, 255, 255), rect, 2)

def count_segment_crossings(routes):
    return _count_segment_crossings(routes)

def count_segment_clearance_conflicts(routes):
    return _count_segment_clearance_conflicts(routes, get_route_diameter, get_required_clearance_mm)

def count_segment_overlaps(routes):
    return _count_segment_overlaps(routes)

def count_ordered_route_turns(route_name, segs):
    return _count_ordered_route_turns(route_name, segs)

def count_solution_turns(routes):
    return _count_solution_turns(routes)

def get_min_piece_length(route_name, terminal_segment=False):
    diameter = get_route_diameter(route_name)
    multiplier = 1.0 if terminal_segment else 2.0
    return diameter * multiplier * min_piece_factor

def merged_route_piece_lengths(route_name, segs):
    return _merged_route_piece_lengths(route_name, segs)

def count_route_short_pieces(route_name, segs):
    return _count_route_short_pieces(route_name, segs, get_min_piece_length)

def count_solution_short_pieces(routes):
    return _count_solution_short_pieces(routes, get_min_piece_length)

def find_route_at_point(routes, world_pt):
    return _find_route_at_point(routes, world_pt, max(40.0, 8.0 / SCALE_PX_PER_MM))

def find_route_hit_at_point(routes, world_pt):
    return _find_route_hit_at_point(routes, world_pt, max(40.0, 8.0 / SCALE_PX_PER_MM))

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
        if shaft_extraction.contains(click_pt) or shaft_extraction.distance(click_pt) < max(40.0, 8.0 / SCALE_PX_PER_MM):
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

def run_sequential_routing(perm, pin_node_map, global_pins, shaft_node_idx, chosen_exhaust_pin, chosen_exhaust_target, shaft_path):
    return _run_sal_sequential_routing(
        perm,
        pin_node_map,
        global_pins,
        shaft_node_idx,
        chosen_exhaust_pin,
        chosen_exhaust_target,
        shaft_path,
        env=current_env,
        machine_angle=machine_angle,
        bend_cost=C_BEND,
        route_start_nodes=get_route_start_nodes,
        route_segments_from_path=_route_segments_from_path,
        run_search=run_super_sink_astar,
        terminal_node_indices=get_all_terminal_node_indices,
        set_terminal_block_weight=set_terminal_block_weight,
        add_route_clearance_weights=add_route_clearance_weights,
        add_route_interaction_weights=add_route_interaction_weights,
        route_diameter=get_route_diameter,
        route_axis_records=_route_axis_records,
    )

def _source_start_nodes(source_spec):
    return _source_start_nodes_for_kd(source_spec, grid_kd)

def _run_pin_min_cost_flow(route_names, target_specs_by_route, terminal_points_by_route, edge_weights=None):
    if not route_names:
        return {}, {}, 0.0, 0
    record_edge_weight_overlay(edge_weights, current_env)

    start_nodes_by_route = {
        route_name: _source_start_nodes(terminal_points_by_route[route_name])
        for route_name in route_names
    }
    network = _build_pin_min_cost_flow_network(
        route_names,
        target_specs_by_route,
        start_nodes_by_route,
        current_env.adj,
        lambda u, v, dist: _weighted_edge_cost(edge_weights, u, v, dist),
        lambda u, v: _line_graph_dir_from_points(current_env, u, v),
        C_BEND,
        OVERLAP_BLOCK_WEIGHT,
    )
    if network is None:
        return None, None, float("inf"), 0
    graph, source, sink, route_flow_nodes = network

    flow, cost = _min_cost_flow(graph, source, sink, len(route_names))
    if flow < len(route_names):
        return None, None, cost, flow

    paths = {}
    targets = {}
    for route_name in route_names:
        path, target = _trace_flow_path(graph, route_flow_nodes[route_name], sink)
        if path is None:
            return None, None, cost, flow
        paths[route_name] = path
        targets[route_name] = target

    return paths, targets, cost, flow

def _run_small_pin_min_cost_flow(room_names, pin_node_map, edge_weights=None):
    target_specs_by_route = _small_pin_target_specs(room_names, pin_node_map)
    terminal_points_by_route = {room_name: get_route_start_nodes(room_name) for room_name in room_names}
    return _run_pin_min_cost_flow(room_names, target_specs_by_route, terminal_points_by_route, edge_weights=edge_weights)

def _route_segments_from_path(route_name, path, pin_name=None, global_pins=None, target=None):
    return _route_segments_from_path_for_env(
        route_name,
        path,
        current_env.nodes,
        add_shaft_entry_segments if shaft_extraction else None,
        pin_name,
        global_pins,
        target,
    )

def _build_routes_from_paths(route_order, paths, targets, global_pins):
    return _build_routes_from_paths_for_env(route_order, paths, targets, global_pins, _route_segments_from_path)

def _route_one_pin_flow(route_name, target_pin, terminal_point, pin_node_map, edge_weights=None):
    target_specs_by_route = {route_name: pin_node_map.get(target_pin, [])}
    terminal_points_by_route = {route_name: terminal_point}
    paths, targets, cost, flow = _run_pin_min_cost_flow(
        [route_name],
        target_specs_by_route,
        terminal_points_by_route,
        edge_weights=edge_weights,
    )
    if flow < 1 or paths is None:
        return None, None, cost
    return paths[route_name], targets[route_name], cost

def _run_large_pin_candidate_search(pin_node_map, shaft_boundary_nodes, edge_weights=None):
    return _search_sal_large_route_candidates(
        pin_node_map, shaft_boundary_nodes, env=current_env, terminals=terminals,
        route_start_nodes=get_route_start_nodes, route_one_pin_flow=_route_one_pin_flow,
        route_segments_from_path=_route_segments_from_path, route_axis_records=_route_axis_records,
        add_route_clearance_weights=add_route_clearance_weights,
        add_route_interaction_weights=add_route_interaction_weights,
        route_diameter=get_route_diameter, count_crossings=count_segment_crossings,
        score_routes=get_solution_score, initial_edge_weights=edge_weights,
    )

def run_small_pin_min_cost_flow_routing(room_names, pin_node_map, global_pins, shaft_node_idx, chosen_exhaust_pin, chosen_exhaust_target, shaft_path):
    base_weights = {}
    prior_axis_records = []
    routes = []

    shaft_segs = _route_segments_from_path("Shaft", shaft_path, chosen_exhaust_pin, global_pins, chosen_exhaust_target)
    routes.append(("Shaft", shaft_segs))
    prior_axis_records.extend(_route_axis_records("Shaft", shaft_segs))
    total_nodes = len(shaft_path)

    kitchen_pin_name = "right_mid" if chosen_exhaust_pin == "left_mid" else "left_mid"
    kitchen_start_nodes = get_route_start_nodes("Kitchen")
    if kitchen_start_nodes:
        kitchen_weights = base_weights.copy()
        add_route_clearance_weights(kitchen_weights, "Kitchen", current_env)
        add_route_interaction_weights(prior_axis_records, get_route_diameter("Kitchen"), kitchen_weights, current_env)
        kitchen_path, _, _, kitchen_target = run_super_sink_astar(
            current_env,
            kitchen_start_nodes,
            [kitchen_pin_name],
            pin_node_map,
            global_pins,
            machine_angle,
            C_BEND,
            edge_weights=kitchen_weights,
        )
        if kitchen_path is None:
            return False, None, "No path to Kitchen", 0
    else:
        return False, None, "Missing Kitchen terminal", 0

    kitchen_segs = _route_segments_from_path("Kitchen", kitchen_path, kitchen_pin_name, global_pins, kitchen_target)
    routes.append(("Kitchen", kitchen_segs))
    prior_axis_records.extend(_route_axis_records("Kitchen", kitchen_segs))
    total_nodes += len(kitchen_path)

    small_weights = base_weights.copy()
    add_static_clearance_weights(small_weights, MACHINE_SMALL_DUCT_D, current_env, allow_shaft_entry=False)
    add_machine_clearance_weights(small_weights, MACHINE_SMALL_DUCT_D, current_env)
    add_route_interaction_weights(prior_axis_records, MACHINE_SMALL_DUCT_D, small_weights, current_env)
    paths, targets, _, flow = _run_small_pin_min_cost_flow(room_names, pin_node_map, edge_weights=small_weights)
    if paths is None:
        return False, None, f"Min-cost flow routed {flow}/{len(room_names)} small ducts", 0

    small_routes, small_nodes = _build_routes_from_paths(room_names, paths, targets, global_pins)
    if small_routes is None:
        return False, None, "Could not build small duct routes", 0
    routes.extend(small_routes)
    total_nodes += small_nodes

    return True, routes, "Success", total_nodes

def _run_two_stage_big_first(room_names, pin_node_map, global_pins, shaft_path):
    large_paths, large_targets, _, large_flow, large_meta = _run_large_pin_candidate_search(
        pin_node_map,
        [shaft_path[0]],
        edge_weights=None,
    )
    if large_paths is None:
        return False, None, f"Min-cost flow routed {large_flow}/2 large ducts", 0

    route_order = ["Shaft", "Kitchen"]
    routes, total_nodes = _build_routes_from_paths(route_order, large_paths, large_targets, global_pins)
    if routes is None:
        return False, None, "Could not build large duct routes", 0

    prior_axis_records = []
    for route_name, segs in routes:
        prior_axis_records.extend(_route_axis_records(route_name, segs))

    small_weights = {}
    add_static_clearance_weights(small_weights, MACHINE_SMALL_DUCT_D, current_env, allow_shaft_entry=False)
    add_machine_clearance_weights(small_weights, MACHINE_SMALL_DUCT_D, current_env)
    add_route_interaction_weights(prior_axis_records, MACHINE_SMALL_DUCT_D, small_weights, current_env)
    small_paths, small_targets, _, flow = _run_small_pin_min_cost_flow(room_names, pin_node_map, edge_weights=small_weights)
    if small_paths is None:
        return False, None, f"Min-cost flow routed {flow}/{len(room_names)} small ducts", 0

    small_routes, small_nodes = _build_routes_from_paths(room_names, small_paths, small_targets, global_pins)
    if small_routes is None:
        return False, None, "Could not build small duct routes", 0
    return True, routes + small_routes, f"big-first {large_meta.get('assignment', '')} {large_meta.get('large_order', '')}", total_nodes + small_nodes

def _run_two_stage_small_first(room_names, pin_node_map, global_pins, shaft_path):
    initial_small_weights = {}
    add_static_clearance_weights(initial_small_weights, MACHINE_SMALL_DUCT_D, current_env, allow_shaft_entry=False)
    add_machine_clearance_weights(initial_small_weights, MACHINE_SMALL_DUCT_D, current_env)
    small_paths, small_targets, _, flow = _run_small_pin_min_cost_flow(room_names, pin_node_map, edge_weights=initial_small_weights)
    if small_paths is None:
        return False, None, f"Min-cost flow routed {flow}/{len(room_names)} small ducts", 0

    small_routes, small_nodes = _build_routes_from_paths(room_names, small_paths, small_targets, global_pins)
    if small_routes is None:
        return False, None, "Could not build small duct routes", 0

    prior_axis_records = []
    for route_name, segs in small_routes:
        prior_axis_records.extend(_route_axis_records(route_name, segs))

    large_weights = {}
    add_static_clearance_weights(large_weights, MACHINE_LARGE_DUCT_D, current_env, allow_shaft_entry=False)
    add_machine_clearance_weights(large_weights, MACHINE_LARGE_DUCT_D, current_env)
    add_route_interaction_weights(prior_axis_records, MACHINE_LARGE_DUCT_D, large_weights, current_env)
    large_paths, large_targets, _, large_flow, large_meta = _run_large_pin_candidate_search(
        pin_node_map,
        [shaft_path[0]],
        edge_weights=large_weights,
    )
    if large_paths is None:
        return False, None, f"Min-cost flow routed {large_flow}/2 large ducts", 0

    large_routes, large_nodes = _build_routes_from_paths(["Shaft", "Kitchen"], large_paths, large_targets, global_pins)
    if large_routes is None:
        return False, None, "Could not build large duct routes", 0
    return True, large_routes + small_routes, f"small-first {large_meta.get('assignment', '')} {large_meta.get('large_order', '')}", large_nodes + small_nodes

def run_two_stage_min_cost_flow_routing(room_names, pin_node_map, global_pins, shaft_path):
    return _select_sal_two_stage_routing(
        lambda: _run_two_stage_big_first(room_names, pin_node_map, global_pins, shaft_path),
        lambda: _run_two_stage_small_first(room_names, pin_node_map, global_pins, shaft_path),
        count_segment_crossings,
        get_solution_score,
    )

def get_solution_score(routes, crossings):
    weights = RouteScoreWeights(
        bend=C_BEND,
        crossing=CROSSING_PENALTY,
        overlap=OVERLAP_SCORE_PENALTY,
        clearance=CLEARANCE_PENALTY,
        short_piece=SHORT_PIECE_SCORE_PENALTY,
    )
    return _score_routes(
        routes,
        weights,
        get_route_diameter,
        get_required_clearance_mm,
        get_min_piece_length,
        crossings=crossings,
    )

def get_route_validation_warnings(routes):
    if not routes:
        return []
    warnings = _route_quality_warnings(
        routes,
        get_route_diameter,
        get_required_clearance_mm,
        get_min_piece_length,
    )
    warnings = _append_allowed_region_warning(warnings, routes, routing_region_base, shaft_extraction)
    if shaft_extraction is not None and DWELLING_SOURCE_MODES[dwelling_source_idx] == "Real DB" and not shaft_core_entry_specs:
        warnings.append("missing core shaft entry metadata")
    return warnings

def get_route_conflict_summary(routes):
    return _route_conflict_summary(
        routes,
        get_route_diameter,
        get_required_clearance_mm,
        get_min_piece_length,
    )

# ──────────────────────────────────────────────────────────────────────────
# TOPOLOGICAL DISTANCE FIELDS AUTO-PLACEMENT ALGORITHMS
# ──────────────────────────────────────────────────────────────────────────
def is_machine_placement_valid(cx, cy, angle):
    global_pins = get_machine_pins(cx, cy, angle)
    return _is_machine_placement_valid_for_placement(
        cx,
        cy,
        global_pins,
        routing_region_base,
        walls,
        columns,
        shafts,
    )

def get_placement_weights():
    return _placement_weights(weight_mode_idx)

def get_auto_placement_scores(env, shaft_boundary_nodes):
    terminal_nodes = {}
    for name, pt in terminals.items():
        _, node_idx = base_regular_kd.query(pt)
        terminal_nodes[name] = int(node_idx)

    return _topological_placement_scores(env, shaft_boundary_nodes, terminal_nodes, get_placement_weights())

def ensure_placement_heatmap_scores():
    global ap_scores, ap_fields
    if ap_scores or base_regular_env is None or base_regular_kd is None or shaft_extraction is None:
        return
    shaft_boundary_nodes, _ = get_shaft_entry_nodes(base_regular_env, base_regular_kd)
    ap_scores, ap_fields = get_auto_placement_scores(base_regular_env, shaft_boundary_nodes)

def _candidate_machine_rooms():
    return _candidate_machine_rooms_for_placement(rooms, MACHINE_OVERALL_W * MACHINE_BODY_H)

def _candidate_room_points(room):
    return _candidate_room_points_for_placement(room, _routing_frame_axes_for_placement())

def _distance_to_allowed_boundary(point):
    if routing_region_base is None:
        return 1e9
    return Point(float(point[0]), float(point[1])).distance(routing_region_base.boundary)

def _core_like_machine_candidate_score(cx, cy, angle, room):
    pins = get_machine_pins(cx, cy, angle)

    shaft_pt = get_representative_point(shaft_extraction) if shaft_extraction else (cx, cy)
    kitchen_pt = terminals.get("Kitchen", (cx, cy))
    return _core_like_machine_candidate_score_for_placement(
        cx,
        cy,
        angle,
        room.polygon,
        pins,
        shaft_pt,
        kitchen_pt,
        "Kitchen" in terminals,
        _distance_to_allowed_boundary,
        _local_axis_to_world,
    )

def _room_field_target_point(room_name):
    room_poly = get_route_room_polygon(room_name)
    if room_poly is None or room_poly.is_empty:
        return terminals.get(room_name)
    centroid = room_poly.centroid
    if room_poly.contains(centroid):
        return (float(centroid.x), float(centroid.y))
    return get_representative_point(room_poly)

def _score_rotation_field_at(cx, cy, angle):
    pins = get_machine_pins(cx, cy, angle)
    shaft_point = get_representative_point(shaft_extraction) if shaft_extraction else None
    return _score_rotation_field_at_for_placement(
        pins,
        angle,
        wet_room_names,
        terminals.keys(),
        shaft_point,
        _room_field_target_point,
        weight_mode_idx,
        _local_axis_to_world,
    )

def apply_field_alignment_rotation():
    global machine_angle, rotation_field_scores
    machine_angle, selected, scores = _select_field_alignment_rotation(
        machine_angle,
        lambda angle: is_machine_placement_valid(machine_cx, machine_cy, angle),
        lambda angle: _score_rotation_field_at(machine_cx, machine_cy, angle),
        ROTATION_FIELD_EPS,
    )
    rotation_field_scores = {
        "H": 0.0 if not math.isfinite(scores.get("H", 0.0)) else float(scores.get("H", 0.0)),
        "V": 0.0 if not math.isfinite(scores.get("V", 0.0)) else float(scores.get("V", 0.0)),
        "selected": selected,
    }
    return selected, scores

def apply_rotation_mode_once():
    if rotation_mode_idx != 1:
        rotation_field_scores.update({"H": 0.0, "V": 0.0, "selected": "Torque"})
        return
    before = machine_angle
    apply_field_alignment_rotation()
    if machine_angle != before:
        pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
        build_grid(machine_pins=pins)

def run_core_workflow_machine_placement():
    global machine_cx, machine_cy, machine_angle, ap_scores, ap_fields
    t0 = time.perf_counter()
    selected, candidate_count = _choose_core_like_machine_placement(
        _candidate_machine_rooms(),
        _candidate_room_points,
        (0, 90, 180, 270),
        is_machine_placement_valid,
        _core_like_machine_candidate_score,
    )
    ap_scores = {}
    ap_fields = {}
    if selected is None:
        return

    best_x, best_y, best_rot, _score = selected
    machine_cx, machine_cy, machine_angle = best_x, best_y, best_rot
    pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
    build_grid(machine_pins=pins)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[Core-like Machine Placement] tried {candidate_count} feasible candidates in {elapsed_ms:.1f}ms")

def run_auto_placement():
    global machine_cx, machine_cy, machine_angle, ap_scores, ap_fields
    if base_regular_env is None or not shaft_extraction:
        return
        
    shaft_boundary_nodes, shaft_node_idx = get_shaft_entry_nodes(base_regular_env, base_regular_kd)
    
    # Topological Distance Fields
    if auto_placement_mode_idx == 1:
        t0 = time.perf_counter()
        
        node_scores, distance_fields = get_auto_placement_scores(base_regular_env, shaft_boundary_nodes)
        ap_scores = node_scores
        ap_fields = distance_fields
        
        if not node_scores:
            return
            
        selected = _choose_topological_machine_placement(
            base_regular_env,
            node_scores,
            distance_fields,
            (0, 90, 180, 270),
            is_machine_placement_valid,
            get_machine_pins,
            lambda pt: int(base_regular_kd.query(pt)[1]),
            wet_room_names,
            get_placement_weights(),
        )
        if selected is not None:
            machine_cx, machine_cy, machine_angle, _score = selected
            pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
            build_grid(machine_pins=pins)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[Auto-Placement] Solved position ({machine_cx}, {machine_cy}) at rotation {machine_angle} in {elapsed_ms:.2f}ms")
            return

    elif auto_placement_mode_idx == 2:
        run_core_workflow_machine_placement()
        return

# ──────────────────────────────────────────────────────────────────────────
# MAIN SOLVER WRAPPER
# ──────────────────────────────────────────────────────────────────────────
def solve_ventilation_routing():
    global edge_weight_debug_map, edge_weight_overlay_excluded_edges
    edge_weight_debug_map = {}
    edge_weight_overlay_excluded_edges = set()
    global_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)

    machine_poly = Polygon([
        global_pins["c_tl"], global_pins["c_tr"], global_pins["c_br"], global_pins["c_bl"]
    ])

    m_center = Point(machine_cx, machine_cy)
    if not routing_region_base or not routing_region_base.contains(m_center):
        return None, "Blocked: Machine outside region", 0.0, 0

    if any(machine_poly.intersects(col) for col in columns):
        return None, "Blocked: Machine collides with column", 0.0, 0

    if grid_nodes is None:
        return None, "Building grid… press Space to retry", 0.0, 0

    if graph_type_idx == 0:
        update_dynamic_env(machine_poly)
    else:
        build_grid(machine_pins=global_pins)
        update_dynamic_env(machine_poly)

    pin_node_map = snap_pins_to_graph(global_pins)
    if not pin_node_map or not shaft_extraction:
        return None, "Blocked: Missing pins or shaft", 0.0, 0

    shaft_boundary_nodes, shaft_node_idx = get_shaft_entry_nodes(current_env, grid_kd)

    # Pre-calculate terminal node indices
    terminal_nodes = get_all_terminal_node_indices(pin_node_map, shaft_node_idx)
    
    # Block other terminals for Shaft search
    shaft_weights = {}
    for edge in _block_terminal_node_edges(shaft_weights, current_env.adj, terminal_nodes, OVERLAP_BLOCK_WEIGHT):
        edge_weight_overlay_excluded_edges.add(edge)
    add_route_clearance_weights(shaft_weights, "Shaft", current_env)

    t_solver = time.perf_counter()

    # 1. Route Shaft via Super Source/Sink
    shaft_path, _, chosen_exhaust_pin, chosen_exhaust_target = run_super_sink_astar(
        current_env,
        shaft_boundary_nodes,
        ["left_mid", "right_mid"],
        pin_node_map,
        global_pins,
        machine_angle,
        C_BEND,
        edge_weights=shaft_weights,
    )

    if shaft_path is None:
        elapsed_ms = (time.perf_counter() - t_solver) * 1000.0
        return None, "Blocked: No path to shaft", elapsed_ms, 0

    # 2. Backtracking search over permutations of the small duct rooms
    from itertools import permutations
    other_rooms = _ordered_small_room_names(terminals, (machine_cx, machine_cy))

    if routing_strategy_idx == 5:
        success, routes_cand, status_cand, total_nodes_cand = run_small_pin_min_cost_flow_routing(
            other_rooms,
            pin_node_map,
            global_pins,
            shaft_node_idx,
            chosen_exhaust_pin,
            chosen_exhaust_target,
            shaft_path,
        )
        elapsed_ms = (time.perf_counter() - t_solver) * 1000.0
        if success:
            status_text = f"Success: Min-cost flow small pins ({get_route_conflict_summary(routes_cand)}) in {elapsed_ms:.1f}ms"
            return routes_cand, status_text, elapsed_ms, total_nodes_cand
        return None, f"Routing Blocked: {status_cand} in {elapsed_ms:.1f}ms", elapsed_ms, 0

    if routing_strategy_idx == 6:
        success, routes_cand, status_cand, total_nodes_cand = run_two_stage_min_cost_flow_routing(
            other_rooms,
            pin_node_map,
            global_pins,
            shaft_path,
        )
        elapsed_ms = (time.perf_counter() - t_solver) * 1000.0
        if success:
            status_text = f"Success: Two-stage MCMF {status_cand} ({get_route_conflict_summary(routes_cand)}) in {elapsed_ms:.1f}ms"
            return routes_cand, status_text, elapsed_ms, total_nodes_cand
        return None, f"Routing Blocked: {status_cand} in {elapsed_ms:.1f}ms", elapsed_ms, 0
    
    if routing_strategy_idx == 0:
        close_to_far = tuple(other_rooms)
        far_to_close = tuple(reversed(other_rooms))
        all_perms = [close_to_far, far_to_close]
    else:
        all_perms = list(permutations(other_rooms))

    best_routes = None
    best_crossings = 1e9
    best_score = 1e18
    best_total_nodes = 0
    perm_attempts = 0

    if routing_strategy_idx in (3, 4):
        # ── Strategy 3 & 4: Negotiated Congestion ──
        nets_list = ["Shaft", "Kitchen"] + other_rooms
        current_paths = {}
        current_pins = {}
        current_pin_targets = {}
        
        P_present = 20000.0
        P_history = 4000.0
        history_congestion = {}
        node_history_congestion = {}
        
        for iteration in range(20):
            perm_attempts += 1
            
            for net_name in nets_list:
                if net_name == "Shaft":
                    start_nodes = shaft_boundary_nodes
                    targets = ["left_mid", "right_mid"]
                else:
                    if net_name == "Kitchen":
                        start_nodes = get_route_start_nodes("Kitchen")
                        if not start_nodes:
                            continue
                        shaft_pin = current_pins.get("Shaft", "left_mid")
                        kitchen_pin_name = "right_mid" if shaft_pin == "left_mid" else "left_mid"
                        targets = [kitchen_pin_name]
                    else:
                        start_nodes = get_route_start_nodes(net_name)
                        if not start_nodes:
                            continue
                        used_small_pins = [current_pins[n] for n in other_rooms if n != net_name and n in current_pins]
                        targets = [p for p in ["tl", "tr", "bl", "br"] if p not in used_small_pins]
                        if not targets:
                            targets = ["tl"]
                
                current_paths[net_name] = None
                
                edge_usage = {}
                node_usage = {}
                for other_name, path in current_paths.items():
                    if path is None:
                        continue
                    for u in path:
                        node_usage[u] = node_usage.get(u, 0) + 1
                    for k in range(len(path) - 1):
                        e = (min(path[k], path[k+1]), max(path[k], path[k+1]))
                        edge_usage[e] = edge_usage.get(e, 0) + 1
                        
                current_weights = {}
                terminal_nodes = get_all_terminal_node_indices(pin_node_map, shaft_node_idx)
                for r_name, t_node_idx in terminal_nodes.items():
                    if r_name == net_name:
                        continue
                    if t_node_idx in current_env.adj:
                        for nbr, _, _ in current_env.adj[t_node_idx]:
                            set_terminal_block_weight(current_weights, t_node_idx, nbr)
                            
                for u in current_env.adj:
                    for v, dist, direction in current_env.adj[u]:
                        edge = (min(u, v), max(u, v))
                        if edge in current_weights and current_weights[edge] >= 1e9:
                            continue
                        
                        pres = edge_usage.get(edge, 0)
                        hist = history_congestion.get(edge, 0.0)
                        
                        node_pres = max(node_usage.get(u, 0), node_usage.get(v, 0))
                        node_hist = max(node_history_congestion.get(u, 0.0), node_history_congestion.get(v, 0.0))
                        
                        congestion_weight = (pres * P_present) + hist + (node_pres * 20000.0) + node_hist
                        if routing_strategy_idx == 4 and net_name in ("Shaft", "Kitchen"):
                            congestion_weight *= 0.35
                        current_weights[edge] = dist + congestion_weight
                add_route_clearance_weights(current_weights, net_name, current_env)

                negotiated_axis_records = []
                for other_name, other_path in current_paths.items():
                    if other_path is None or other_name == net_name:
                        continue
                    other_segs = _route_segments_from_path(other_name, other_path)
                    negotiated_axis_records.extend(_route_axis_records(other_name, other_segs))
                add_route_interaction_weights(
                    negotiated_axis_records,
                    get_route_diameter(net_name),
                    current_weights,
                    current_env,
                )
                        
                path, _, chosen_pin, chosen_target = run_super_sink_astar(
                    current_env,
                    start_nodes,
                    targets,
                    pin_node_map,
                    global_pins,
                    machine_angle,
                    C_BEND,
                    edge_weights=current_weights,
                )
                
                if path is not None:
                    current_paths[net_name] = path
                    current_pins[net_name] = chosen_pin
                    current_pin_targets[net_name] = chosen_target
                    
            routes_cand = []
            success = True
            total_nodes_cand = 0
            for name in nets_list:
                path = current_paths.get(name)
                if path is None:
                    success = False
                    break
                segs = []
                if name == "Shaft" and path and shaft_extraction:
                    add_shaft_entry_segments(segs, path[0])
                for k in range(len(path) - 1):
                    p1 = current_env.nodes[path[k]]
                    p2 = current_env.nodes[path[k+1]]
                    segs.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))
                if name in current_pins:
                    add_port_stub_segment(segs, current_pins[name], path[-1], global_pins, current_pin_targets.get(name))
                routes_cand.append((name, segs))
                total_nodes_cand += len(path)
                
            if success:
                crossings = count_segment_crossings(routes_cand)
                score = get_solution_score(routes_cand, crossings)
                
                if score < best_score:
                    best_score = score
                    best_crossings = crossings
                    best_routes = routes_cand
                    best_total_nodes = total_nodes_cand
                    
                if crossings == 0:
                    elapsed_ms = (time.perf_counter() - t_solver) * 1000.0
                    status_text = f"Success: Routed all (tried {perm_attempts} iters, {get_route_conflict_summary(routes_cand)}) in {elapsed_ms:.1f}ms"
                    return routes_cand, status_text, elapsed_ms, total_nodes_cand
                    
                edge_counts = {}
                node_counts = {}
                for name, path in current_paths.items():
                    if path is None:
                        continue
                    for u in path:
                        node_counts[u] = node_counts.get(u, 0) + 1
                    for k in range(len(path) - 1):
                        edge = (min(path[k], path[k+1]), max(path[k], path[k+1]))
                        edge_counts[edge] = edge_counts.get(edge, 0) + 1
                        
                for edge, count in edge_counts.items():
                    if count > 1:
                        history_congestion[edge] = history_congestion.get(edge, 0.0) + P_history
                for node, count in node_counts.items():
                    if count > 1:
                        node_history_congestion[node] = node_history_congestion.get(node, 0.0) + 4000.0
                        
        if best_routes is not None:
            elapsed_ms = (time.perf_counter() - t_solver) * 1000.0
            status_text = f"Success: Routed all (tried {perm_attempts} iters, {get_route_conflict_summary(best_routes)}) in {elapsed_ms:.1f}ms"
            return best_routes, status_text, elapsed_ms, best_total_nodes
        else:
            elapsed_ms = (time.perf_counter() - t_solver) * 1000.0
            return None, f"Routing Blocked (tried {perm_attempts} iters) in {elapsed_ms:.1f}ms", elapsed_ms, 0

    for perm in all_perms:
        perm_attempts += 1
        success, routes_cand, status_cand, total_nodes_cand = run_sequential_routing(
            perm, pin_node_map, global_pins, shaft_node_idx, chosen_exhaust_pin, chosen_exhaust_target, shaft_path
        )
        if success:
            crossings = count_segment_crossings(routes_cand)
            score = get_solution_score(routes_cand, crossings)
            if score < best_score:
                best_score = score
                best_crossings = crossings
                best_routes = routes_cand
                best_total_nodes = total_nodes_cand
            if routing_strategy_idx == 1 and crossings == 0:
                break

    if best_routes is not None:
        elapsed_ms = (time.perf_counter() - t_solver) * 1000.0
        status_text = f"Success: Routed all (tried {perm_attempts} perms, {get_route_conflict_summary(best_routes)}) in {elapsed_ms:.1f}ms"
        return best_routes, status_text, elapsed_ms, best_total_nodes
    else:
        elapsed_ms = (time.perf_counter() - t_solver) * 1000.0
        return None, f"Routing Blocked (tried {perm_attempts} perms) in {elapsed_ms:.1f}ms", elapsed_ms, 0

# ──────────────────────────────────────────────────────────────────────────
# DWELLING AND ROOM GENERATORS
# ──────────────────────────────────────────────────────────────────────────
def generate_synthetic_dwelling():
    global rooms, columns, shafts, covers, doors, walls, wall_polys, routing_region_base, shaft_extraction, terminals, wet_room_names
    global machine_cx, machine_cy, machine_angle, _bnd_segs, hannan_static_cache
    global current_scenario_label, current_scenario_summary, shaft_core_entry_specs, shaft_entry_geometry_by_node
    
    rooms_m = generative_layout.generate_layout(width=15.0, height=11.0, num_rooms=8)
    
    rooms = []
    covered_names = ["Hallway", "Kitchen", "Bathroom", "Bathroom 1", "Bathroom 2", "Toilet", "Washroom", "Bedroom 1"]
    for r in rooms_m:
        scaled_poly = snap_to_integer_grid(shapely_scale(r.polygon, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0)))
        room_scaled = generative_layout.Room(scaled_poly, r.name)
        room_scaled.has_cover = any(cn in r.name for cn in covered_names)
        rooms.append(room_scaled)
    covers = [r.polygon for r in rooms if getattr(r, "has_cover", False)]
        
    shafts_m = generative_layout.generate_mep_shafts(rooms_m)
    shafts = [snap_to_integer_grid(shapely_scale(s, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0))) for s in shafts_m]

    columns_m = generative_layout.generate_structural_grid(unary_union([r.polygon for r in rooms_m]), spacing=4.0)
    columns = []
    for col in columns_m:
        col_scaled = snap_to_integer_grid(shapely_scale(col, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0)))
        if any(col_scaled.intersects(s) for s in shafts):
            continue
        columns.append(col_scaled)
    
    doors_m = generative_layout.find_door_openings(rooms_m)
    doors = []
    for d in doors_m:
        d_scaled = {
            "d1": (round(d["d1"][0] * SCALE_TO_MM), round(d["d1"][1] * SCALE_TO_MM)),
            "d2": (round(d["d2"][0] * SCALE_TO_MM), round(d["d2"][1] * SCALE_TO_MM)),
            "swing_dir": d["swing_dir"],
            "width": d["width"] * SCALE_TO_MM,
            "is_entrance": d.get("is_entrance", False)
        }
        doors.append(d_scaled)
        
    entrance = generative_layout.find_entrance_door(rooms_m, unary_union([r.polygon for r in rooms_m]))
    if entrance:
        doors.append({
            "d1": (round(entrance["d1"][0] * SCALE_TO_MM), round(entrance["d1"][1] * SCALE_TO_MM)),
            "d2": (round(entrance["d2"][0] * SCALE_TO_MM), round(entrance["d2"][1] * SCALE_TO_MM)),
            "swing_dir": entrance["swing_dir"],
            "width": entrance["width"] * SCALE_TO_MM,
            "is_entrance": True
        })

    # Extract wall line centerlines and subtract columns/shafts
    walls_m = []
    for i in range(len(rooms_m)):
        for j in range(i + 1, len(rooms_m)):
            shared = rooms_m[i].polygon.intersection(rooms_m[j].polygon)
            if isinstance(shared, LineString):
                walls_m.append(shared)
            elif hasattr(shared, 'geoms'):
                for g in shared.geoms:
                    if isinstance(g, LineString):
                        walls_m.append(g)
                        
    raw_walls = [snap_to_integer_grid(shapely_scale(w, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0))) for w in walls_m]
    walls = []
    for rw in raw_walls:
        w_cut = rw
        for col in columns:
            w_cut = w_cut.difference(col)
        for s in shafts:
            w_cut = w_cut.difference(s)
        if w_cut.is_empty:
            continue
        if w_cut.geom_type == 'LineString':
            walls.append(w_cut)
        elif hasattr(w_cut, 'geoms'):
            for g in w_cut.geoms:
                if g.geom_type == 'LineString':
                    walls.append(g)
    
    wall_polys = []
    for w in walls:
        wp = w.buffer(WALL_THICKNESS / 2 - 0.1)
        for col in columns:
            wp = wp.difference(col)
        for s in shafts:
            wp = wp.difference(s)
        if not wp.is_empty:
            wall_polys.append(wp)
    
    routing_region_m = unary_union([r.polygon for r in rooms_m if any(cn in r.name for cn in covered_names)])
    routing_region_base = snap_to_integer_grid(shapely_scale(routing_region_m, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0)))
    
    for col in columns:
        routing_region_base = routing_region_base.difference(col)
    for shaft in shafts:
        routing_region_base = routing_region_base.difference(shaft)
        
    shaft_extraction = shafts[0] if shafts else None
    
    wet_rooms = [r for r in rooms if any(w in r.name for w in ["Kitchen", "Bathroom", "Toilet", "Washroom"])]
    terminals = {}
    for r in wet_rooms:
        t_pt = get_representative_point(r.polygon)
        terminals[r.name] = t_pt
        
    best_room = None
    best_dist = 1e9
    if shaft_extraction:
        rep_pt = shaft_extraction.representative_point()
        sx, sy = rep_pt.x, rep_pt.y
        for r in wet_rooms:
            if any(w in r.name for w in ["Bathroom", "Washroom"]):
                rx, ry = r.polygon.centroid.x, r.polygon.centroid.y
                dist = abs(sx - rx) + abs(sy - ry)
                if dist < best_dist:
                    best_dist = dist
                    best_room = r
                    
    if best_room:
        machine_cx, machine_cy = get_representative_point(best_room.polygon)
    else:
        machine_cx, machine_cy = 7500.0, 5500.0
        
    machine_angle = 0
    wet_room_names = list(terminals.keys())
    rebuild_wet_room_outer_accents()
    current_scenario_label = "synthetic"
    current_scenario_summary = {}
    shaft_core_entry_specs = []
    shaft_entry_geometry_by_node = {}
    _bnd_segs = None
    hannan_static_cache = {}
    build_base_regular_grid()
    pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
    build_grid(machine_pins=pins)

def _iter_lines(geom):
    if geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            yield line
    elif hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from _iter_lines(part)

def _cut_line_obstacles(line):
    cut = line
    for col in columns:
        cut = cut.difference(col)
    for shaft in shafts:
        cut = cut.difference(shaft)
    return list(_iter_lines(cut))

def _derive_real_walls():
    derived = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            shared = rooms[i].polygon.boundary.intersection(rooms[j].polygon.boundary)
            for line in _iter_lines(shared):
                if line.length > 50:
                    derived.extend(_cut_line_obstacles(line))
    return [line for line in derived if line.length > 50]

def _build_wall_polys():
    polys = []
    for w in walls:
        wp = w.buffer(WALL_THICKNESS / 2 - 0.1)
        for col in columns:
            wp = wp.difference(col)
        for s in shafts:
            wp = wp.difference(s)
        if not wp.is_empty:
            polys.append(wp)
    return polys

def _choose_initial_machine_position():
    if not terminals:
        return (7500.0, 5500.0)
    if not shaft_extraction:
        return next(iter(terminals.values()))

    sx, sy = get_representative_point(shaft_extraction)
    candidates = []
    for name, pt in terminals.items():
        priority = 0 if any(key in name for key in ["Bathroom", "Washroom", "Toilet"]) else 1
        candidates.append((priority, abs(sx - pt[0]) + abs(sy - pt[1]), pt))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]

def rebuild_wet_room_outer_accents():
    global wet_room_outer_accents
    wet_names = set(wet_room_names)
    accents = []
    for room in rooms:
        if getattr(room, "name", None) not in wet_names:
            continue
        poly = getattr(room, "polygon", None)
        if poly is None or poly.is_empty:
            continue
        accent = poly.buffer(10.0, join_style=2)
        if not accent.is_empty:
            accents.append(accent)
    wet_room_outer_accents = accents

def _load_real_dwelling():
    global current_scenario_label, current_scenario_summary
    if load_dwelling_scenario is None:
        raise RuntimeError("dwelling_export.demo_loader could not be imported.")
    if not REAL_DWELLING_DB.exists():
        raise RuntimeError(f"Dwelling database not found: {REAL_DWELLING_DB}")

    execution, dwelling_id = REAL_DWELLING_SCENARIOS[real_scenario_idx % len(REAL_DWELLING_SCENARIOS)]
    scenario = load_dwelling_scenario(
        db_path=REAL_DWELLING_DB,
        execution=execution,
        dwelling_id=dwelling_id,
        scale_to_mm=True,
        frame_name=ROUTING_FRAME_OPTIONS[routing_frame_idx % len(ROUTING_FRAME_OPTIONS)],
        preferred_shaft_installation=PREFERRED_SHAFT_INSTALLATION,
    )
    current_scenario_label = f"{execution} / {dwelling_id}"
    current_scenario_summary = scenario_summary(scenario) if scenario_summary else {}
    return scenario

def generate_new_dwelling():
    global rooms, columns, shafts, covers, doors, walls, wall_polys, routing_region_base, shaft_extraction, terminals, wet_room_names
    global machine_cx, machine_cy, machine_angle, _bnd_segs, hannan_static_cache
    global shaft_core_entry_specs, shaft_entry_geometry_by_node
    global preferred_terminal_points_by_room, preferred_terminal_areas

    preferred_terminal_points_by_room = {}
    preferred_terminal_areas = []
    invalidate_room_start_node_cache()
    if DWELLING_SOURCE_MODES[dwelling_source_idx] == "Random Synthetic":
        generate_synthetic_dwelling()
        return

    try:
        scenario = _load_real_dwelling()
    except Exception as err:
        print(f"Real dwelling load failed, falling back to synthetic: {err}")
        generate_synthetic_dwelling()
        return

    rooms = scenario.rooms
    columns = list(scenario.columns)
    shafts = list(scenario.shafts)
    covers = list(getattr(scenario, "covers", []) or [room.polygon for room in rooms if getattr(room, "has_cover", False)])
    shaft_extraction = scenario.shaft_extraction
    routing_region_base = scenario.routing_region_base
    terminals = dict(scenario.terminals)
    wet_room_names = list(terminals.keys())
    rebuild_wet_room_outer_accents()
    doors = []
    walls = _derive_real_walls()
    wall_polys = _build_wall_polys()
    shaft_core_entry_specs = _build_core_shaft_entry_specs(scenario)
    shaft_entry_geometry_by_node = {}

    machine_cx, machine_cy = _choose_initial_machine_position()
    machine_angle = 0
    _bnd_segs = None
    hannan_static_cache = {}
    build_base_regular_grid()
    run_auto_placement()
    pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
    build_grid(machine_pins=pins)

def get_turbo_color(t):
    return _turbo_color(t)

def get_viridis_color(t):
    return _viridis_color(t)

def get_heatmap_color(t):
    return _heatmap_color(t, heatmap_palette_idx)

def draw_colorbar(screen, node_scores):
    if not node_scores:
        return
        
    cb_x = COLORBAR_LEFT + (COLORBAR_W - 20) // 2
    cb_y = CANVAS_TOP + 40
    cb_w = 20
    cb_h = CANVAS_H - 150
    
    # Border
    pygame.draw.rect(screen, (255, 255, 255), (cb_x - 1, cb_y - 1, cb_w + 2, cb_h + 2), 1)
    
    # Fill gradient: High cost at the top (y=0, t=1.0), Low cost at the bottom (y=cb_h, t=0.0)
    for y in range(cb_h):
        t = 1.0 - (y / cb_h)
        if heatmap_scale_mode == 0:
            t_sat = min(1.0, t / 0.75)
            c = get_heatmap_color(t_sat)
        else:
            c = get_heatmap_color(t)
        pygame.draw.line(screen, c, (cb_x, cb_y + y), (cb_x + cb_w - 1, cb_y + y))
        
    font_lbl = pygame.font.SysFont("Outfit", 14, bold=True)
    
    high_color = get_heatmap_color(1.0)
    low_color = get_heatmap_color(0.0)
    lbl_high = font_lbl.render("HIGH", True, high_color)
    screen.blit(lbl_high, (cb_x + cb_w // 2 - lbl_high.get_width() // 2, cb_y - 18))
    
    lbl_low = font_lbl.render("LOW", True, low_color)
    screen.blit(lbl_low, (cb_x + cb_w // 2 - lbl_low.get_width() // 2, cb_y + cb_h + 6))
    
    palette_name = "VIRIDIS" if heatmap_palette_idx == 1 else "TURBO"
    lbl_title = font_lbl.render(palette_name, True, COLOR_TEXT)
    screen.blit(lbl_title, (cb_x + cb_w // 2 - lbl_title.get_width() // 2, cb_y + cb_h + 24))

def _score_to_heatmap_t(score, min_s, max_s):
    return _score_to_heatmap_t_value(score, min_s, max_s, heatmap_scale_mode)

def _interpolate_regular_score(wx, wy, score_grid):
    gx = wx / GRID_SPACING
    gy = wy / GRID_SPACING
    ix0 = math.floor(gx)
    iy0 = math.floor(gy)
    fx = gx - ix0
    fy = gy - iy0

    q00 = score_grid.get((ix0, iy0))
    q10 = score_grid.get((ix0 + 1, iy0))
    q01 = score_grid.get((ix0, iy0 + 1))
    q11 = score_grid.get((ix0 + 1, iy0 + 1))

    if q00 is not None and q10 is not None and q01 is not None and q11 is not None:
        return (
            q00 * (1.0 - fx) * (1.0 - fy) +
            q10 * fx * (1.0 - fy) +
            q01 * (1.0 - fx) * fy +
            q11 * fx * fy
        )

    candidates = []
    for ix, iy, score in (
        (ix0, iy0, q00),
        (ix0 + 1, iy0, q10),
        (ix0, iy0 + 1, q01),
        (ix0 + 1, iy0 + 1, q11),
    ):
        if score is None:
            continue
        dx = wx - ix * GRID_SPACING
        dy = wy - iy * GRID_SPACING
        d2 = dx * dx + dy * dy
        candidates.append((d2, score))

    if not candidates:
        return None

    d2, score = min(candidates, key=lambda item: item[0])
    if d2 <= (GRID_SPACING * 1.45) ** 2:
        return score
    return None

def _build_heatmap_surface(node_scores):
    min_s = min(node_scores.values())
    max_s = max(node_scores.values())
    score_grid = {}
    for node_idx, score in node_scores.items():
        if node_idx >= len(base_regular_env.nodes):
            continue
        x, y = base_regular_env.nodes[node_idx]
        score_grid[(round(float(x) / GRID_SPACING), round(float(y) / GRID_SPACING))] = float(score)

    low_w = 320
    low_h = max(1, round(low_w * CANVAS_H / CANVAS_W))
    low = pygame.Surface((low_w, low_h), pygame.SRCALPHA)
    alpha = 150

    for py in range(low_h):
        abs_y = CANVAS_TOP + (py + 0.5) * CANVAS_H / low_h
        wy = 11000.0 - (abs_y - OFFSET_Y) / SCALE_PX_PER_MM
        for px in range(low_w):
            abs_x = CANVAS_LEFT + (px + 0.5) * CANVAS_W / low_w
            wx = (abs_x - OFFSET_X) / SCALE_PX_PER_MM
            score = _interpolate_regular_score(wx, wy, score_grid)
            if score is None:
                continue
            c = get_heatmap_color(_score_to_heatmap_t(score, min_s, max_s))
            low.set_at((px, py), (c[0], c[1], c[2], alpha))

    return pygame.transform.smoothscale(low, (CANVAS_W, CANVAS_H))

def draw_distance_heatmap(screen, node_scores):
    if not node_scores or base_regular_env is None:
        return
    key = (
        id(base_regular_env),
        id(node_scores),
        len(node_scores),
        min(node_scores.values()),
        max(node_scores.values()),
        heatmap_scale_mode,
        heatmap_palette_idx,
        CANVAS_W,
        CANVAS_H,
    )
    if heatmap_surface_cache["key"] != key:
        heatmap_surface_cache["surface"] = _build_heatmap_surface(node_scores)
        heatmap_surface_cache["key"] = key
    screen.blit(heatmap_surface_cache["surface"], (CANVAS_LEFT, CANVAS_TOP))

def _cool_colormap(t):
    return _cool_colormap_value(t)

def _edge_weight_log_scale():
    finite_values = [v for v in edge_weight_debug_map.values() if v < OVERLAP_BLOCK_WEIGHT]
    max_ratio = max(finite_values) if finite_values else 1.0
    return max_ratio, math.log1p(max(max_ratio, 1.0))

def draw_edge_weight_heatmap(screen):
    if not edge_weight_heatmap_enabled or not edge_weight_debug_map or current_env is None:
        return
    _, log_max = _edge_weight_log_scale()

    for (u, v), ratio in edge_weight_debug_map.items():
        if u not in current_env.adj or u >= len(current_env.nodes) or v >= len(current_env.nodes):
            continue
        p1 = to_screen(current_env.nodes[u][0], current_env.nodes[u][1])
        p2 = to_screen(current_env.nodes[v][0], current_env.nodes[v][1])
        if ratio >= OVERLAP_BLOCK_WEIGHT:
            color = COLOR_BLOCKED_EDGE
            width = 5
        else:
            t = math.log1p(max(0.0, ratio)) / log_max
            color = _cool_colormap(t)
            width = 3
        pygame.draw.line(screen, color, p1, p2, width)

def draw_edge_weight_colorbar(screen):
    if not edge_weight_heatmap_enabled or not edge_weight_debug_map:
        return

    cb_x = COLORBAR_LEFT + (COLORBAR_W - 20) // 2
    cb_y = CANVAS_TOP + 40
    cb_w = 20
    cb_h = CANVAS_H - 150
    max_ratio, log_max = _edge_weight_log_scale()

    pygame.draw.rect(screen, (255, 255, 255), (cb_x - 1, cb_y - 1, cb_w + 2, cb_h + 2), 1)
    for y in range(cb_h):
        t = 1.0 - (y / cb_h)
        c = _cool_colormap(t)
        pygame.draw.line(screen, c, (cb_x, cb_y + y), (cb_x + cb_w - 1, cb_y + y))

    font_lbl = pygame.font.SysFont("Outfit", 13, bold=True)
    lbl_title = font_lbl.render("WGT", True, COLOR_TEXT)
    screen.blit(lbl_title, (cb_x + cb_w // 2 - lbl_title.get_width() // 2, cb_y - 32))

    lbl_high = font_lbl.render(f"+{max_ratio:.1f}x", True, _cool_colormap(1.0))
    screen.blit(lbl_high, (cb_x + cb_w // 2 - lbl_high.get_width() // 2, cb_y - 16))
    lbl_low = font_lbl.render("+0x", True, _cool_colormap(0.0))
    screen.blit(lbl_low, (cb_x + cb_w // 2 - lbl_low.get_width() // 2, cb_y + cb_h + 6))

    if any(v >= OVERLAP_BLOCK_WEIGHT for v in edge_weight_debug_map.values()):
        block_rect = pygame.Rect(cb_x, cb_y + cb_h + 28, cb_w, 8)
        pygame.draw.rect(screen, COLOR_BLOCKED_EDGE, block_rect)
        lbl_block = font_lbl.render("BLOCK", True, COLOR_TEXT)
        screen.blit(lbl_block, (cb_x + cb_w // 2 - lbl_block.get_width() // 2, cb_y + cb_h + 40))

def get_route_draw_width(route_name):
    if route_real_diameter_width_enabled:
        return max(1, int(round(get_route_diameter(route_name) * SCALE_PX_PER_MM)))
    return 5 if route_name in LARGE_DUCT_ROUTE_NAMES else 3

def record_history(routes, crossings_count, elapsed_ms):
    """Append one sample to the history buffers (called after every successful solve)."""
    global hist_sample_count
    length_m = 0.0
    turns = 0
    if routes:
        for _, segs in routes:
            length_m += sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in segs)
        turns = count_solution_turns(routes)
    length_m /= 1000.0  # mm → m
    score = get_solution_score(routes, crossings_count) if routes else 0
    turns_per_m = turns / length_m if length_m > 0 else 0.0
    
    hist_length.append(length_m)
    hist_score.append(score)
    hist_turns.append(turns)
    hist_turns_per_len.append(turns_per_m)
    hist_exec_ms.append(elapsed_ms)
    hist_sample_count += 1
    if routes:
        update_auto_best_logs(routes, "Auto best", elapsed_ms, 0)

def clear_history_buffers():
    global hist_sample_count
    hist_length.clear()
    hist_score.clear()
    hist_turns.clear()
    hist_turns_per_len.clear()
    hist_exec_ms.clear()
    hist_event_markers.clear()
    hist_sample_count = 0

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
    return {
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
        "preferred_terminal_points_by_room": {
            room_name: [tuple(pt) for pt in points]
            for room_name, points in preferred_terminal_points_by_room.items()
        },
        "preferred_terminal_areas": [
            {"room": area["room"], "bounds": tuple(area["bounds"])}
            for area in preferred_terminal_areas
        ],
        "status": status,
        "total_nodes": int(total_nodes),
        "kpis": kpis,
    }

def log_current_solution(routes, status, elapsed_ms, total_nodes):
    global selected_log_id
    if not routes:
        return False
    record_history(routes, count_segment_crossings(routes), elapsed_ms)
    log_id = len(solution_logs) + 1
    entry = snapshot_current_state(routes, status, elapsed_ms, total_nodes)
    entry["id"] = log_id
    entry["hist_idx"] = hist_sample_count - 1 if hist_length else None
    entry["kind"] = "manual"
    solution_logs.append(entry)
    selected_log_id = log_id
    if hist_length:
        hist_event_markers.append((hist_sample_count - 1, f"L{log_id}", (255, 255, 255)))
    return True

def _metric_value_for_log(entry, metric):
    return _metric_value_for_log_entry(entry, metric)

def _replace_hist_marker(label, idx, color):
    global hist_event_markers
    hist_event_markers = [marker for marker in hist_event_markers if marker[1] != label]
    hist_event_markers.append((idx, label, color))

def update_auto_best_logs(routes, status, elapsed_ms, total_nodes):
    if not routes:
        return
    entry_base = snapshot_current_state(routes, status, elapsed_ms, total_nodes)
    hist_idx = hist_sample_count - 1 if hist_length else None
    if hist_idx is None:
        return
    for metric, entry, label, color in _best_log_updates(entry_base, hist_idx, auto_best_logs):
        auto_best_logs[metric] = entry
        _replace_hist_marker(label, hist_idx, color)

def restore_solution_log(log_entry):
    global machine_cx, machine_cy, machine_angle
    global graph_type_idx, routing_strategy_idx, router_backend_idx, heuristic_mode_idx, room_start_mode_idx
    global rotation_mode_idx
    global weight_mode_idx, edge_weight_view_mode_idx, route_real_diameter_width_enabled, min_piece_factor
    global C_BEND, crossing_penalty_multiplier
    global preferred_terminal_points_by_room, preferred_terminal_areas, selected_log_id

    machine_cx, machine_cy, machine_angle = log_entry["machine"]
    graph_type_idx = log_entry["graph_type_idx"]
    routing_strategy_idx = log_entry["routing_strategy_idx"]
    router_backend_idx = log_entry["router_backend_idx"]
    heuristic_mode_idx = log_entry["heuristic_mode_idx"]
    rotation_mode_idx = log_entry.get("rotation_mode_idx", 0)
    room_start_mode_idx = log_entry["room_start_mode_idx"]
    weight_mode_idx = log_entry["weight_mode_idx"]
    edge_weight_view_mode_idx = log_entry["edge_weight_view_mode_idx"]
    route_real_diameter_width_enabled = log_entry["route_real_diameter_width_enabled"]
    min_piece_factor = log_entry["min_piece_factor"]
    C_BEND = log_entry.get("bend_weight", C_BEND_DEFAULT)
    crossing_penalty_multiplier = log_entry.get("crossing_penalty_multiplier", CROSSING_MULTIPLIER_DEFAULT)
    refresh_route_weight_constants()
    preferred_terminal_points_by_room = {
        room_name: [tuple(pt) for pt in points]
        for room_name, points in log_entry["preferred_terminal_points_by_room"].items()
    }
    preferred_terminal_areas = [
        {"room": area["room"], "bounds": tuple(area["bounds"])}
        for area in log_entry.get("preferred_terminal_areas", [])
    ]
    selected_log_id = log_entry["id"]
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
        window_width=WINDOW_WIDTH,
        window_height=WINDOW_HEIGHT,
        panel_width=PANEL_W,
        solution_logs=solution_logs,
        auto_best_logs=auto_best_logs,
        selected_log_id=selected_log_id,
        plot_background_color=COLOR_PLOT_BG,
        text_color=COLOR_TEXT,
        muted_color=COLOR_MUTED,
    )

def draw_plots(screen, font_small, font_bold):
    """Draw routing metrics and solve-time sparklines in the right panel."""
    px = WINDOW_WIDTH - PANEL_W + 8
    pw = PANEL_W - 24
    ph = 104   # height of each chart area
    gap = 8   # gap between the plots
    titles   = ["DUCT LENGTH  (m)", "COST SCORE", "TURNS", "TURNS / METRE", "SOLVER TIME (ms)"]
    buffers  = [hist_length, hist_score, hist_turns, hist_turns_per_len, hist_exec_ms]
    colors   = [(46, 204, 113), (241, 196, 15), (155, 89, 182), (26, 188, 156), (52, 152, 219)]
    y_starts = [50 + i * (ph + gap) for i in range(len(buffers))]

    for title, buf, col, py in zip(titles, buffers, colors, y_starts):
        # Background
        pygame.draw.rect(screen, COLOR_PLOT_BG, (px - 4, py, pw + 8, ph), border_radius=6)
        pygame.draw.rect(screen, (55, 55, 70),  (px - 4, py, pw + 8, ph), 1, border_radius=6)

        # Title
        lbl = font_bold.render(title, True, col)
        screen.blit(lbl, (px, py + 6))

        chart_y  = py + 26
        chart_h  = ph - 42
        chart_w  = pw
        n = len(buf)

        if n < 2:
            lbl_wait = font_small.render("Move machine to trace…", True, COLOR_MUTED)
            screen.blit(lbl_wait, (px, chart_y + chart_h // 2 - 8))
            continue

        vals  = list(buf)
        # Zero-based scaling
        lo    = 0.0
        hi    = max(vals)
        span  = hi - lo if hi > lo else 1.0

        def sx(i):  return px + int(i / (n - 1) * chart_w)
        def sy(v):  return chart_y + chart_h - int((v - lo) / span * chart_h)

        # Highlight minimum values reached
        min_val = min(vals)
        min_idx = vals.index(min_val)
        min_y = sy(min_val)
        
        # Draw horizontal dotted line at minimum
        for dash_x in range(px, px + chart_w, 6):
            pygame.draw.line(screen, (231, 76, 60),
                             (dash_x, min_y), (min(dash_x + 3, px + chart_w), min_y))

        log_markers_to_draw = []
        visible_start_idx = hist_sample_count - n
        # Draw vertical event markers (strategy changes, weight mode changes, etc.)
        for idx, label, m_col in hist_event_markers:
            rel = idx - visible_start_idx
            if 0 <= rel < n:
                mx_px = sx(rel)
                lbl_ev = font_small.render(label, True, m_col)
                is_log_marker = label.startswith("L") or label.startswith("Best")
                if is_log_marker:
                    value_idx = max(0, min(len(vals) - 1, rel))
                    my_px = sy(vals[value_idx])
                    log_markers_to_draw.append((mx_px, my_px, label, m_col))
                else:
                    for dash_y in range(chart_y, chart_y + chart_h, 8):
                        pygame.draw.line(screen, m_col, (mx_px, dash_y), (mx_px, min(dash_y + 4, chart_y + chart_h)), 1)
                    screen.blit(lbl_ev, (mx_px - lbl_ev.get_width() // 2, py + 16))

        # Sparkline
        pts = [(sx(i), sy(v)) for i, v in enumerate(vals)]
        if len(pts) >= 2:
            pygame.draw.lines(screen, col, False, pts, 2)

        # Draw minimum dot
        pygame.draw.circle(screen, (231, 76, 60), (sx(min_idx), min_y), 4)

        # Current value dot
        pygame.draw.circle(screen, (255, 255, 255), pts[-1], 4)
        pygame.draw.circle(screen, col, pts[-1], 3)

        for mx_px, my_px, label, m_col in log_markers_to_draw:
            diamond = [
                (mx_px, my_px - 6),
                (mx_px + 6, my_px),
                (mx_px, my_px + 6),
                (mx_px - 6, my_px),
            ]
            pygame.draw.polygon(screen, m_col, diamond)
            pygame.draw.polygon(screen, (255, 255, 255), diamond, 1)
            lbl_ev = font_small.render(label, True, m_col)
            screen.blit(lbl_ev, (mx_px + 8, max(chart_y, my_px - 8)))

        # Percentages compare the current value to the best observed value.
        def get_worse_than_min_str(v):
            if abs(min_val) < 1e-5:
                return "0.0%"
            pct = ((v - min_val) / min_val) * 100.0
            return f"+{max(0.0, pct):.1f}%"

        cur_val = vals[-1]
        lbl_cur = font_small.render(f"Cur: {cur_val:.1f} ({get_worse_than_min_str(cur_val)})", True, (255, 255, 255))
        lbl_min = font_small.render(f"Min: {min_val:.1f}", True, (231, 76, 60))

        screen.blit(lbl_cur, (px, chart_y + chart_h + 2))
        screen.blit(lbl_min, (px + chart_w - lbl_min.get_width(), chart_y + chart_h + 2))

HELP_TEXT = {
    "auto": [
        "[A] Auto-placement on/off",
        "[P] Cycle placement mode",
        "[U] Rotation mode",
        "[V] Placement heatmap",
        "[H] Heatmap scale",
        "[B] Heatmap palette",
        "[W] Placement weights",
    ],
    "solver": [
        "[C] Routing strategy",
        "[L] Router backend",
        "[Y] A* heuristic",
        "[Tab] Grid type",
        "[G] Grid mesh",
        "[T] Start mode",
        "Terminal: click nearest node",
        "Term. area: drag rectangle",
        "Ctrl removes preferences",
        "Reset prefs clears all",
        "Sliders: piece/bend/cross",
        "[M] Edge weights",
        "[N] Small/big weight view",
        "[X] Real pipe width",
    ],
    "machine": [
        "Drag machine with mouse",
        "Wheel: rotate machine",
        "[U] Torque/field rotation",
        "Shift+wheel: zoom",
        "Shift+drag or middle drag: pan",
        "[D] Dwelling source",
        "[O] Routing frame",
    ],
    "kpi": [
        "Plots compare current values",
        "against best observed minimum.",
        "Solver time excludes rendering.",
    ],
    "status": [
        "Routing status and solver time.",
        "Right log panel stores",
        "session-local states.",
        "[Esc] Clear selection / ruler",
        "[Space] New apartment",
    ],
}

def draw_card_help_button(screen, card_id, rect, font_small):
    global help_button_rects
    btn = pygame.Rect(rect.right - 26, rect.y + 8, 18, 18)
    help_button_rects[card_id] = btn
    active = help_popup_card == card_id
    fill = (58, 80, 94) if active else (50, 55, 66)
    pygame.draw.rect(screen, fill, btn, border_radius=9)
    pygame.draw.rect(screen, COLOR_MUTED, btn, 1, border_radius=9)
    lbl = font_small.render("?", True, COLOR_TEXT)
    screen.blit(lbl, (btn.centerx - lbl.get_width() // 2, btn.centery - lbl.get_height() // 2))

def draw_help_popup(screen, font_small):
    if not help_popup_card or help_popup_card not in HELP_TEXT:
        return
    lines = HELP_TEXT[help_popup_card]
    width = 235
    line_h = 18
    height = 18 + len(lines) * line_h
    rect = pygame.Rect(CANVAS_LEFT + 16, CANVAS_TOP + 58, width, height)
    pygame.draw.rect(screen, (22, 22, 30), rect, border_radius=6)
    pygame.draw.rect(screen, (120, 130, 145), rect, 1, border_radius=6)
    for i, line in enumerate(lines):
        lbl = font_small.render(line, True, COLOR_TEXT)
        screen.blit(lbl, (rect.x + 10, rect.y + 10 + i * line_h))

def set_transient_message(text, duration_ms=2400):
    global transient_message, transient_message_until_ms
    transient_message = str(text)
    transient_message_until_ms = pygame.time.get_ticks() + int(duration_ms)

def draw_transient_message(screen, font_small):
    if not transient_message or pygame.time.get_ticks() > transient_message_until_ms:
        return
    surf = font_small.render(transient_message, True, COLOR_TEXT)
    rect = pygame.Rect(CANVAS_LEFT + 16, CANVAS_TOP + 16, surf.get_width() + 20, surf.get_height() + 12)
    pygame.draw.rect(screen, (55, 45, 35), rect, border_radius=5)
    pygame.draw.rect(screen, (241, 196, 15), rect, 1, border_radius=5)
    screen.blit(surf, (rect.left + 10, rect.top + 6))

def draw_viewer_legend(screen, font_small):
    x = CANVAS_LEFT + 18
    y = CANVAS_TOP + CANVAS_H - 34

    if terminal_validity_overlay_enabled:
        allowed_label = font_small.render("allowed", True, COLOR_PLAN_LABEL)
        blocked_label = font_small.render("blocked", True, COLOR_PLAN_LABEL)
        width = 34 + allowed_label.get_width() + 28 + blocked_label.get_width() + 18
        rect = pygame.Rect(x, y - 30, width, 24)
        pygame.draw.rect(screen, (248, 247, 243), rect, border_radius=4)
        pygame.draw.rect(screen, (140, 146, 150), rect, 1, border_radius=4)
        draw_terminal_validity_square(screen, (rect.x + 15, rect.centery), 12, True)
        screen.blit(allowed_label, (rect.x + 28, rect.centery - allowed_label.get_height() // 2))
        blocked_x = rect.x + 34 + allowed_label.get_width() + 22
        draw_terminal_validity_square(screen, (blocked_x, rect.centery), 12, False)
        screen.blit(blocked_label, (blocked_x + 14, rect.centery - blocked_label.get_height() // 2))

    label = font_small.render("wet rooms", True, COLOR_PLAN_LABEL)
    rect = pygame.Rect(x, y, label.get_width() + 58, 24)
    pygame.draw.rect(screen, (248, 247, 243), rect, border_radius=4)
    pygame.draw.rect(screen, (140, 146, 150), rect, 1, border_radius=4)
    line_y = rect.centery
    pygame.draw.line(screen, COLOR_WET_ROOM_ACCENT, (rect.x + 10, line_y), (rect.x + 36, line_y), 3)
    pygame.draw.line(screen, COLOR_WALL, (rect.x + 10, line_y + 3), (rect.x + 36, line_y + 3), 1)
    screen.blit(label, (rect.x + 44, rect.centery - label.get_height() // 2))

def main():
    global machine_cx, machine_cy, machine_angle, show_grid_graph, graph_type_idx, routing_strategy_idx
    global router_backend_idx, heuristic_mode_idx, auto_placement_mode_idx, show_heatmap, hist_ap_idx, weight_mode_idx, ap_scores, ap_fields, heatmap_scale_mode, heatmap_palette_idx
    global real_scenario_idx, routing_frame_idx, dwelling_source_idx, room_start_mode_idx
    global edge_weight_heatmap_enabled, edge_weight_view_mode_idx, route_real_diameter_width_enabled
    global rotation_mode_idx
    global view_pan_x_px, view_pan_y_px
    global zoom_level
    global help_popup_card
    global min_piece_factor, is_fullscreen
    global preferred_terminal_tool_mode
    global selected_log_id
    
    pygame.init()
    pygame.font.init()
    
    update_window_layout(WINDOW_WIDTH, WINDOW_HEIGHT)
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Integrated Auto-Placement & Ventilation Router (Demo 10.8)")
    clock = pygame.time.Clock()
    
    font_title = pygame.font.SysFont("Outfit", 24, bold=True)
    font_bold = pygame.font.SysFont("Outfit", 18, bold=True)
    font_small = pygame.font.SysFont("Outfit", 15)
    
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
    last_wheel_rotate_ms = 0
    
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
                hist_event_markers.append((hist_sample_count - 1, "Auto", (230, 126, 34)))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE and not is_fullscreen:
                update_window_layout(event.w, event.h)
                screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    clicked_help = False
                    for card_id, rect in help_button_rects.items():
                        if rect.collidepoint((mx, my)):
                            help_popup_card = None if help_popup_card == card_id else card_id
                            clicked_help = True
                            break
                    if clicked_help:
                        continue
                    if min_piece_slider_rect.collidepoint((mx, my)):
                        dragging_min_piece_slider = True
                        set_min_piece_factor_from_slider_x(mx)
                        routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                        if routes and not status.startswith("Blocked"):
                            record_current_solution(routes, elapsed_ms, f"Min:{min_piece_factor:.2f}", (241, 196, 15))
                        continue
                    if bend_weight_slider_rect.collidepoint((mx, my)):
                        dragging_bend_weight_slider = True
                        set_bend_weight_from_slider_x(mx)
                        routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                        if routes and not status.startswith("Blocked"):
                            record_current_solution(routes, elapsed_ms, f"B:{C_BEND:.0f}", (155, 89, 182))
                        continue
                    if bend_weight_reset_rect.collidepoint((mx, my)):
                        reset_bend_weight()
                        routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                        if routes and not status.startswith("Blocked"):
                            record_current_solution(routes, elapsed_ms, "B:reset", (155, 89, 182))
                        continue
                    if crossing_weight_slider_rect.collidepoint((mx, my)):
                        dragging_crossing_weight_slider = True
                        set_crossing_weight_from_slider_x(mx)
                        routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                        if routes and not status.startswith("Blocked"):
                            record_current_solution(routes, elapsed_ms, f"X:{crossing_penalty_multiplier:.1f}", (230, 126, 34))
                        continue
                    if crossing_weight_reset_rect.collidepoint((mx, my)):
                        reset_crossing_weight()
                        routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                        if routes and not status.startswith("Blocked"):
                            record_current_solution(routes, elapsed_ms, "X:reset", (230, 126, 34))
                        continue
                    tool_action = handle_canvas_tool_button_click((mx, my))
                    if tool_action:
                        if tool_action == "ruler":
                            ruler_mode = not ruler_mode
                            ruler_dragging = False
                            if ruler_mode:
                                preferred_terminal_tool_mode = None
                            set_ruler_cursor(ruler_mode)
                        elif tool_action == "weights":
                            edge_weight_heatmap_enabled = not edge_weight_heatmap_enabled
                        elif tool_action == "weight_view":
                            edge_weight_view_mode_idx = (edge_weight_view_mode_idx + 1) % 2
                        continue
                    log_action = handle_solution_log_click((mx, my))
                    if log_action == "log":
                        log_current_solution(routes, status, elapsed_ms, total_nodes)
                        continue
                    elif log_action is not None:
                        if isinstance(log_action, str) and log_action.startswith("best:"):
                            log_entry = auto_best_logs.get(log_action.split(":", 1)[1])
                        else:
                            log_entry = next((entry for entry in solution_logs if entry["id"] == log_action), None)
                        if log_entry is not None:
                            routes, status, elapsed_ms, total_nodes = restore_solution_log(log_entry)
                            if routes and not status.startswith("Blocked"):
                                record_current_solution(routes, elapsed_ms, f"Back:L{log_action}", (255, 255, 255))
                        continue
                    terminal_tool_action = handle_terminal_tool_button_click((mx, my))
                    if terminal_tool_action:
                        if terminal_tool_action == "reset":
                            preferred_terminal_points_by_room.clear()
                            preferred_terminal_areas.clear()
                            routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                            if routes and not status.startswith("Blocked"):
                                record_current_solution(routes, elapsed_ms, "Prefs reset", (26, 188, 156))
                        if preferred_terminal_tool_mode:
                            ruler_mode = False
                            ruler_dragging = False
                        set_ruler_cursor(bool(preferred_terminal_tool_mode))
                        continue
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        panning_view = True
                        pan_last_pos = (mx, my)
                        try:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEALL)
                        except pygame.error:
                            pass
                        continue
                    world_x, world_y = to_mm(mx, my)
                    if ruler_mode:
                        ruler_dragging = True
                        ruler_start_mm = (world_x, world_y)
                        ruler_end_mm = (world_x, world_y)
                        continue
                    if preferred_terminal_tool_mode == "point":
                        remove_marker = bool(pygame.key.get_mods() & pygame.KMOD_CTRL)
                        changed, marker_room = apply_preferred_terminal_point((world_x, world_y), remove=remove_marker)
                        if marker_room:
                            selected_route_name = marker_room
                        elif not remove_marker:
                            set_transient_message("Invalid terminal: too close to wall or outside allowed room buffer")
                        if changed:
                            routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                            if routes and not status.startswith("Blocked"):
                                label = "Term-" if remove_marker else "Term+"
                                record_current_solution(routes, elapsed_ms, label, (26, 188, 156))
                        continue
                    if preferred_terminal_tool_mode == "area":
                        terminal_area_dragging = True
                        terminal_area_remove = bool(pygame.key.get_mods() & pygame.KMOD_CTRL)
                        terminal_area_start_mm = (world_x, world_y)
                        terminal_area_end_mm = (world_x, world_y)
                        continue
                    g_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
                    m_poly = Polygon([g_pins["c_tl"], g_pins["c_tr"], g_pins["c_br"], g_pins["c_bl"]])
                    p_obj = Point(world_x, world_y)
                    
                    if m_poly.contains(p_obj) or m_poly.distance(p_obj) < 200.0:
                        dragging = True
                        auto_placement_mode_idx = 0
                        drag_offset_x = world_x - machine_cx
                        drag_offset_y = world_y - machine_cy
                        continue
                    route_names = {name for name, _ in routes} if routes else set()
                    room_hit = find_room_route_at_point((world_x, world_y), route_names)
                    route_hit = find_route_hit_at_point(routes, (world_x, world_y))
                    direct_duct_click_mm = max(20.0, 4.0 / SCALE_PX_PER_MM)
                    if route_hit and (not room_hit or route_hit[1] <= direct_duct_click_mm):
                        selected_route_name = route_hit[0]
                        continue
                    if room_hit:
                        selected_route_name = room_hit
                        continue
                    selected_route_name = None
                elif event.button == 2:
                    panning_view = True
                    pan_last_pos = event.pos
                    try:
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEALL)
                    except pygame.error:
                        pass
                elif event.button == 4: # Scroll Up (CCW)
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        zoom_at_screen_point(zoom_level * 1.12, event.pos)
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
                        hist_event_markers.append((hist_sample_count - 1, f"Rot:{machine_angle}", (46, 204, 113)))
                elif event.button == 5: # Scroll Down (CW)
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        zoom_at_screen_point(zoom_level / 1.12, event.pos)
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
                        hist_event_markers.append((hist_sample_count - 1, f"Rot:{machine_angle}", (46, 204, 113)))
                        
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 2):
                    if event.button == 1 and terminal_area_dragging:
                        changed, marker_room = apply_preferred_terminal_area(
                            terminal_area_start_mm,
                            terminal_area_end_mm,
                            remove=terminal_area_remove,
                        )
                        if marker_room:
                            selected_route_name = marker_room
                        if changed:
                            routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                            if routes and not status.startswith("Blocked"):
                                label = "Area-" if terminal_area_remove else "Area+"
                                record_current_solution(routes, elapsed_ms, label, (155, 89, 182))
                    dragging = False
                    dragging_min_piece_slider = False
                    dragging_bend_weight_slider = False
                    dragging_crossing_weight_slider = False
                    ruler_dragging = False
                    terminal_area_dragging = False
                    panning_view = False
                    pan_last_pos = None
                    set_ruler_cursor(ruler_mode or bool(preferred_terminal_tool_mode))
                    
            elif event.type == pygame.MOUSEMOTION:
                if dragging_min_piece_slider:
                    set_min_piece_factor_from_slider_x(event.pos[0])
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        record_current_solution(routes, elapsed_ms)
                elif dragging_bend_weight_slider:
                    set_bend_weight_from_slider_x(event.pos[0])
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        record_current_solution(routes, elapsed_ms)
                elif dragging_crossing_weight_slider:
                    set_crossing_weight_from_slider_x(event.pos[0])
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        record_current_solution(routes, elapsed_ms)
                elif panning_view and pan_last_pos is not None:
                    mx, my = event.pos
                    dx = mx - pan_last_pos[0]
                    dy = my - pan_last_pos[1]
                    view_pan_x_px += dx
                    view_pan_y_px += dy
                    pan_last_pos = (mx, my)
                    update_view_transform()
                elif ruler_dragging:
                    ruler_end_mm = to_mm(event.pos[0], event.pos[1])
                elif terminal_area_dragging:
                    terminal_area_end_mm = to_mm(event.pos[0], event.pos[1])
                elif dragging:
                    mx, my = event.pos
                    wx, wy = to_mm(mx, my)
                    machine_cx = wx - drag_offset_x
                    machine_cy = wy - drag_offset_y
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
                        update_window_layout(screen.get_width(), screen.get_height())
                    else:
                        update_window_layout(1700, 930)
                        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)

                elif event.key == pygame.K_ESCAPE:
                    if ruler_mode:
                        ruler_mode = False
                        ruler_dragging = False
                        set_ruler_cursor(False)
                    elif preferred_terminal_tool_mode:
                        preferred_terminal_tool_mode = None
                        terminal_area_dragging = False
                        set_ruler_cursor(False)
                    else:
                        selected_route_name = None
                elif event.key == pygame.K_SPACE:
                    if DWELLING_SOURCE_MODES[dwelling_source_idx] == "Real DB":
                        real_scenario_idx = (real_scenario_idx + 1) % len(REAL_DWELLING_SCENARIOS)
                    generate_new_dwelling()
                    solution_logs.clear()
                    auto_best_logs.clear()
                    selected_log_id = None
                    hist_length.clear()
                    hist_score.clear()
                    hist_turns.clear()
                    hist_turns_per_len.clear()
                    hist_exec_ms.clear()
                    hist_event_markers.clear()
                    if auto_placement_mode_idx > 0:
                        needs_auto_placement = True
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                    
                elif event.key == pygame.K_r:
                    auto_placement_mode_idx = 0
                    machine_angle = (machine_angle + 90) % 360
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        hist_event_markers.append((hist_sample_count - 1, f"Rot:{machine_angle}", (46, 204, 113)))
                    
                elif event.key == pygame.K_g:
                    show_grid_graph = not show_grid_graph
                    
                elif event.key == pygame.K_c:
                    routing_strategy_idx = (routing_strategy_idx + 1) % len(ROUTING_STRATEGIES)
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)

                elif event.key == pygame.K_d:
                    dwelling_source_idx = (dwelling_source_idx + 1) % len(DWELLING_SOURCE_MODES)
                    generate_new_dwelling()
                    solution_logs.clear()
                    auto_best_logs.clear()
                    selected_log_id = None
                    hist_length.clear()
                    hist_score.clear()
                    hist_turns.clear()
                    hist_turns_per_len.clear()
                    hist_exec_ms.clear()
                    hist_event_markers.clear()
                    if auto_placement_mode_idx > 0:
                        needs_auto_placement = True
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        hist_event_markers.append((hist_sample_count - 1, f"Src:{dwelling_source_idx}", (52, 152, 219)))

                elif event.key == pygame.K_o:
                    routing_frame_idx = (routing_frame_idx + 1) % len(ROUTING_FRAME_OPTIONS)
                    generate_new_dwelling()
                    solution_logs.clear()
                    auto_best_logs.clear()
                    selected_log_id = None
                    if auto_placement_mode_idx > 0:
                        needs_auto_placement = True
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        hist_event_markers.append((hist_sample_count - 1, f"Frame:{routing_frame_idx}", (230, 126, 34)))

                elif event.key == pygame.K_t:
                    room_start_mode_idx = (room_start_mode_idx + 1) % len(ROOM_START_MODES)
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        hist_event_markers.append((hist_sample_count - 1, f"Start:{room_start_mode_idx}", (155, 89, 182)))
                        hist_event_markers.append((hist_sample_count - 1, f"Strat:{routing_strategy_idx}", (52, 152, 219)))

                elif event.key == pygame.K_l:
                    router_backend_idx = (router_backend_idx + 1) % len(ROUTER_BACKENDS)
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        hist_event_markers.append((hist_sample_count - 1, f"R:{router_backend_idx}", (230, 126, 34)))

                elif event.key == pygame.K_y:
                    heuristic_mode_idx = (heuristic_mode_idx + 1) % len(HEURISTIC_MODES)
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        hist_event_markers.append((hist_sample_count - 1, f"Heur:{heuristic_mode_idx}", (241, 196, 15)))
                    
                elif event.key == pygame.K_TAB:
                    graph_type_idx = (graph_type_idx + 1) % len(GRAPH_TYPES)
                    g_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
                    build_grid(machine_pins=g_pins)
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        hist_event_markers.append((hist_sample_count - 1, f"Grid:{graph_type_idx}", (155, 89, 182)))
                    
                elif event.key == pygame.K_a:
                    if auto_placement_mode_idx > 0:
                        auto_placement_mode_idx = 0
                    else:
                        auto_placement_mode_idx = 2
                    if auto_placement_mode_idx > 0:
                        needs_auto_placement = True
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    
                elif event.key == pygame.K_p:
                    auto_placement_mode_idx = (auto_placement_mode_idx + 1) % len(AUTO_PLACEMENT_MODES)
                    if auto_placement_mode_idx > 0:
                        needs_auto_placement = True
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()

                elif event.key == pygame.K_u:
                    rotation_mode_idx = (rotation_mode_idx + 1) % len(ROTATION_MODES)
                    apply_rotation_mode_once()
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        hist_event_markers.append((hist_sample_count - 1, f"RotMode:{rotation_mode_idx}", (95, 178, 218)))
                    
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
                        hist_event_markers.append((hist_sample_count - 1, f"H:{'Log' if heatmap_scale_mode==1 else 'Lin'}", (150, 150, 150)))

                elif event.key == pygame.K_b:
                    heatmap_palette_idx = (heatmap_palette_idx + 1) % 2
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        hist_event_markers.append((hist_sample_count - 1, f"Pal:{'Vir' if heatmap_palette_idx==1 else 'Tur'}", (26, 188, 156)))
                    
                elif event.key == pygame.K_w:
                    weight_mode_idx = (weight_mode_idx + 1) % 2
                    if auto_placement_mode_idx > 0:
                        needs_auto_placement = True
                    else:
                        if base_regular_env is not None and shaft_extraction is not None:
                            shaft_boundary_nodes, _ = get_shaft_entry_nodes(base_regular_env, base_regular_kd)
                            node_scores, distance_fields = get_auto_placement_scores(base_regular_env, shaft_boundary_nodes)
                            ap_scores = node_scores
                            ap_fields = distance_fields
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c, elapsed_ms)
                        hist_event_markers.append((hist_sample_count - 1, f"W:{'Eq' if weight_mode_idx==1 else 'Def'}", (241, 196, 15)))
                    
        # ── RENDERING ────────────────────────────────────────────────────────
        screen.fill(COLOR_BG)
        active_selection = bool(selected_route_name and routes and any(name == selected_route_name for name, _ in routes))
        if not active_selection:
            selected_route_name = None
        selected_room_poly = get_route_room_polygon(selected_route_name) if selected_route_name else None
        
        for room in rooms:
            if not hasattr(room, 'polygon') or room.polygon.is_empty:
                continue
            coords = list(room.polygon.exterior.coords)
            screen_coords = [to_screen(x, y) for x, y in coords]
            room_name = getattr(room, "name", None)
            is_selected_room = (
                selected_route_name
                and (
                    room_name == selected_route_name
                    or (selected_room_poly is not None and room.polygon.equals(selected_room_poly))
                )
            )
            if selected_route_name and not is_selected_room:
                color = COLOR_DESELECTED_ROOM
            else:
                color = COLOR_ROOM_COVERED if room.has_cover else COLOR_ROOM
            pygame.draw.polygon(screen, color, screen_coords)

        draw_geometry_overlay(screen, covers, COLOR_COVER_OVERLAY)

        if show_heatmap:
            ensure_placement_heatmap_scores()
            if ap_scores:
                draw_distance_heatmap(screen, ap_scores)
                if not edge_weight_heatmap_enabled:
                    draw_colorbar(screen, ap_scores)

        if edge_weight_heatmap_enabled:
            refresh_edge_weight_view_overlay(routes)
        draw_edge_weight_heatmap(screen)
        draw_edge_weight_colorbar(screen)
        draw_terminal_validity_overlay(screen)

        draw_wet_room_outer_accents(screen)

        for room in rooms:
            if not hasattr(room, 'polygon') or room.polygon.is_empty:
                continue
            coords = list(room.polygon.exterior.coords)
            screen_coords = [to_screen(x, y) for x, y in coords]
            pygame.draw.polygon(screen, COLOR_WALL, screen_coords, WALL_DRAW_WIDTH)
            
        for d in doors:
            # Draw door line segment
            sp1 = to_screen(d["d1"][0], d["d1"][1])
            sp2 = to_screen(d["d2"][0], d["d2"][1])
            pygame.draw.line(screen, COLOR_DOOR, sp1, sp2, 4)
            
        for col_poly in columns:
            coords = list(col_poly.exterior.coords)
            screen_coords = [to_screen(x, y) for x, y in coords]
            pygame.draw.polygon(screen, COLOR_COLUMN, screen_coords)
            
        for s_poly in shafts:
            coords = list(s_poly.exterior.coords)
            screen_coords = [to_screen(x, y) for x, y in coords]
            is_active_shaft = shaft_extraction is not None and s_poly.equals(shaft_extraction)
            shaft_color = COLOR_SHAFT if is_active_shaft else COLOR_SHAFT_INACTIVE
            pygame.draw.polygon(screen, shaft_color, screen_coords)
            if not is_active_shaft:
                draw_polygon_hatch(screen, s_poly, COLOR_SHAFT_INACTIVE_HATCH, spacing=9)
                pygame.draw.polygon(screen, COLOR_WALL, screen_coords, 1)
            
        if show_grid_graph and current_env is not None:
            for u in current_env.adj:
                for v, dist, direction in current_env.adj[u]:
                    if u < v:
                        p1 = current_env.nodes[u]
                        p2 = current_env.nodes[v]
                        sp1 = to_screen(p1[0], p1[1])
                        sp2 = to_screen(p2[0], p2[1])
                        pygame.draw.line(screen, COLOR_GRAPH_EDGE, sp1, sp2, 1)
            for p in current_env.nodes:
                pygame.draw.circle(screen, COLOR_GRAPH_NODE, to_screen(p[0], p[1]), 2)
            
        for r_name, pt in terminals.items():
            s_pt = to_screen(pt[0], pt[1])
            c_core = ROUTE_COLORS.get(r_name, (255, 255, 255))
            if selected_route_name and r_name != selected_route_name:
                c_core = COLOR_DESELECTED_PIN
                ring_color = (70, 74, 78)
                text_color = (84, 88, 94)
            else:
                ring_color = (255, 255, 255)
                text_color = COLOR_PLAN_LABEL
            pygame.draw.circle(screen, ring_color, s_pt, 7)
            pygame.draw.circle(screen, c_core, s_pt, 5)
            lbl_name = r_name.replace("Bathroom", "Bath").replace("Washroom", "Wash")
            text_surf = font_small.render(lbl_name, True, text_color)
            draw_outlined_text(
                screen,
                font_small,
                lbl_name,
                (s_pt[0] - text_surf.get_width() // 2, s_pt[1] + 10),
                text_color,
            )
            
        if shaft_extraction:
            s_rep = get_representative_point(shaft_extraction)
            s_pt = to_screen(s_rep[0], s_rep[1])
            pygame.draw.circle(screen, (255, 255, 255), s_pt, 8)
            pygame.draw.circle(screen, (231, 76, 60), s_pt, 6)
            
        if routes:
            for name, segs in routes:
                width = get_route_draw_width(name)
                if selected_route_name == name:
                    for p1, p2 in segs:
                        sp1 = to_screen(p1[0], p1[1])
                        sp2 = to_screen(p2[0], p2[1])
                        pygame.draw.line(screen, COLOR_SELECTION_HALO, sp1, sp2, width + 6)
                c = ROUTE_COLORS.get(name, COLOR_TEXT)
                if selected_route_name and selected_route_name != name:
                    c = COLOR_DESELECTED_ROUTE
                for p1, p2 in segs:
                    sp1 = to_screen(p1[0], p1[1])
                    sp2 = to_screen(p2[0], p2[1])
                    pygame.draw.line(screen, c, sp1, sp2, width)

        draw_preferred_terminal_areas(screen, selected_route_name)
        draw_routed_terminal_endpoint_markers(screen, routes, selected_route_name)
        draw_preferred_terminal_markers(screen, selected_route_name, routes)
        draw_terminal_area_drag(screen, terminal_area_start_mm, terminal_area_end_mm if terminal_area_dragging else None)
                    
        # Bounding box coordinates for machine
        g_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
        m_screen_pts = [
            to_screen(g_pins["c_tl"][0], g_pins["c_tl"][1]),
            to_screen(g_pins["c_tr"][0], g_pins["c_tr"][1]),
            to_screen(g_pins["c_br"][0], g_pins["c_br"][1]),
            to_screen(g_pins["c_bl"][0], g_pins["c_bl"][1])
        ]
        
        base_color = (230, 126, 34) if auto_placement_mode_idx > 0 else (127, 140, 141)
        pygame.draw.polygon(screen, base_color, m_screen_pts)
        pygame.draw.polygon(screen, (255, 255, 255), m_screen_pts, 2)

        selected_pins = get_selected_pin_names(selected_route_name, routes, g_pins) if selected_route_name else set()
        
        for pin_name in ["tl", "tr", "bl", "br", "left_mid", "right_mid"]:
            pt = g_pins[pin_name]
            sp = to_screen(pt[0], pt[1])
            is_large = pin_name in ("left_mid", "right_mid")
            color = (241, 196, 15) if is_large else (230, 126, 34)
            ring_color = (255, 255, 255)
            if selected_route_name and pin_name not in selected_pins:
                color = COLOR_DESELECTED_PIN
                ring_color = (80, 84, 88)
            size = 5 if is_large else 4
            pygame.draw.circle(screen, color, sp, size)
            pygame.draw.circle(screen, ring_color, sp, size, 1)
            
        if auto_placement_mode_idx == 1 and ap_fields:
            s_rep = get_representative_point(shaft_extraction)
            _, p_left_idx = grid_kd.query(g_pins["left_mid"])
            _, p_right_idx = grid_kd.query(g_pins["right_mid"])
            d_left = ap_fields["Shaft"].get(int(p_left_idx), 1e9)
            d_right = ap_fields["Shaft"].get(int(p_right_idx), 1e9)
            exhaust_pt = g_pins["left_mid"] if d_left < d_right else g_pins["right_mid"]
            kitchen_pin = "right_mid" if d_left < d_right else "left_mid"
            
            sp_port = to_screen(exhaust_pt[0], exhaust_pt[1])
            sp_term = to_screen(s_rep[0], s_rep[1])
            pygame.draw.line(screen, (46, 204, 113), sp_port, sp_term, 2)
            
            if "Kitchen" in terminals:
                k_term = terminals["Kitchen"]
                k_port = g_pins[kitchen_pin]
                pygame.draw.line(screen, (241, 196, 15), to_screen(k_port[0], k_port[1]), to_screen(k_term[0], k_term[1]), 2)
                
            small_pins = ["tl", "tr", "bl", "br"]
            remaining_rooms = [r for r in wet_room_names if r != "Kitchen"]
            used_pins = set()
            for r_name in remaining_rooms:
                term_pt = terminals[r_name]
                best_d = 1e9
                best_p = None
                for p in small_pins:
                    if p in used_pins: continue
                    _, p_idx = grid_kd.query(g_pins[p])
                    d = ap_fields[r_name].get(int(p_idx), 1e9)
                    if d < best_d:
                        best_d = d
                        best_p = p
                if best_p:
                    used_pins.add(best_p)
                    port_pt = g_pins[best_p]
                    pygame.draw.line(screen, ROUTE_COLORS.get(r_name, COLOR_TEXT), to_screen(port_pt[0], port_pt[1]), to_screen(term_pt[0], term_pt[1]), 1)

        # ── SIDEBAR PANEL ──
        draw_ruler_overlay(screen, font_small, ruler_start_mm, ruler_end_mm)
        draw_canvas_tool_controls(screen, font_small, ruler_mode)
        draw_terminal_tool_buttons(screen, font_bold, font_small)

        pygame.draw.rect(screen, COLOR_PANEL, (0, 0, CANVAS_LEFT - 10, WINDOW_HEIGHT))
        pygame.draw.line(screen, COLOR_WALL, (CANVAS_LEFT - 10, 0), (CANVAS_LEFT - 10, WINDOW_HEIGHT), 2)
        
        title_surf = font_title.render("Auto-Placement visualizer", True, COLOR_TEXT)
        screen.blit(title_surf, (20, 20))
        sub_surf = font_small.render("Vents & Extraction Router Dashboard", True, COLOR_MUTED)
        screen.blit(sub_surf, (20, 42))
        
        # 1. Auto-placement State Card
        auto_card = pygame.Rect(15, 75, CANVAS_LEFT - 40, 135)
        pygame.draw.rect(screen, (40, 45, 55), auto_card, border_radius=6)
        lbl_ap_title = font_bold.render("AUTO-PLACEMENT STATE", True, COLOR_TEXT)
        screen.blit(lbl_ap_title, (25, 85))
        draw_card_help_button(screen, "auto", auto_card, font_small)
        mode_text = AUTO_PLACEMENT_MODES[auto_placement_mode_idx]
        lbl_ap_mode = font_bold.render(f"Mode: {mode_text}", True, COLOR_TEXT)
        screen.blit(lbl_ap_mode, (25, 105))
        lbl_ap_keys = font_small.render("[P] Mode | [V] Heatmap", True, COLOR_MUTED)
        screen.blit(lbl_ap_keys, (25, 125))
        h_text = "Disabled"
        if show_heatmap:
            scale_text = "Linear" if heatmap_scale_mode == 0 else "Log"
            palette_text = "Viridis" if heatmap_palette_idx == 1 else "Turbo"
            h_text = f"{palette_text} / {scale_text}"
        lbl_ap_heatmap = font_small.render(f"[V] Heatmap: {h_text}", True, COLOR_MUTED)
        screen.blit(lbl_ap_heatmap, (25, 145))
        lbl_ap_scale = font_small.render("[A] Auto | [?] More", True, COLOR_MUTED)
        screen.blit(lbl_ap_scale, (25, 160))
        w_text = "Default" if weight_mode_idx == 0 else "Equal (1.0)"
        lbl_ap_weights = font_small.render(f"[W] Placement Weights: {w_text}", True, COLOR_MUTED)
        screen.blit(lbl_ap_weights, (25, 180))
        rot_mode_short = "Field" if rotation_mode_idx == 1 else "Torque"
        lbl_ap_rot = font_small.render(f"[U] Rotation: {rot_mode_short}", True, COLOR_MUTED)
        screen.blit(lbl_ap_rot, (25, 195))
        
        # 2. Solver Config Card
        solver_card = pygame.Rect(15, 220, CANVAS_LEFT - 40, 250)
        pygame.draw.rect(screen, (40, 45, 55), solver_card, border_radius=6)
        lbl_solv_title = font_bold.render("ROUTING PATH SOLVER", True, COLOR_TEXT)
        screen.blit(lbl_solv_title, (25, 230))
        draw_card_help_button(screen, "solver", solver_card, font_small)
        lbl_strat = font_small.render(f"Strategy: {ROUTING_STRATEGIES[routing_strategy_idx]}", True, COLOR_TEXT)
        screen.blit(lbl_strat, (25, 250))
        lbl_router = font_small.render(f"Router: {ROUTER_BACKENDS[router_backend_idx]}", True, COLOR_TEXT)
        screen.blit(lbl_router, (25, 270))
        lbl_heur = font_small.render(f"Heuristic: {HEURISTIC_MODES[heuristic_mode_idx]}", True, COLOR_TEXT)
        screen.blit(lbl_heur, (25, 290))
        lbl_graph = font_small.render(f"Grid type: {GRAPH_TYPES[graph_type_idx]}", True, COLOR_TEXT)
        screen.blit(lbl_graph, (25, 310))
        lbl_start = font_small.render(f"Starts: {ROOM_START_MODES[room_start_mode_idx]}", True, COLOR_TEXT)
        screen.blit(lbl_start, (25, 330))
        mw_mode = "On" if edge_weight_heatmap_enabled else "Off"
        diameter_mode = "Real" if route_real_diameter_width_enabled else "Fixed"
        weight_view = "Small" if edge_weight_view_mode_idx == 0 else "Big"
        lbl_mw = font_small.render(f"Edge weights: {mw_mode}/{weight_view} | Pipes: {diameter_mode}", True, COLOR_TEXT)
        screen.blit(lbl_mw, (25, 350))
        selected_text = selected_route_name if selected_route_name else "None"
        preferred_count = sum(len(points) for points in preferred_terminal_points_by_room.values())
        lbl_selected = font_small.render(f"Selected: {selected_text[:14]} | Prefs: {preferred_count}", True, COLOR_TEXT)
        screen.blit(lbl_selected, (25, 365))
        draw_min_piece_slider(screen, font_small, 25, 385, CANVAS_LEFT - 70)
        draw_weight_slider(screen, font_small, 25, 420, CANVAS_LEFT - 70, "Bend weight", C_BEND, C_BEND_MIN, C_BEND_MAX, (155, 89, 182), "bend", integer=True)
        draw_weight_slider(screen, font_small, 25, 452, CANVAS_LEFT - 70, "Cross x bend", crossing_penalty_multiplier, CROSSING_MULTIPLIER_MIN, CROSSING_MULTIPLIER_MAX, (230, 126, 34), "crossing", "x")
        
        # 3. Placement Info Card
        machine_card = pygame.Rect(15, 480, CANVAS_LEFT - 40, 105)
        pygame.draw.rect(screen, (40, 45, 55), machine_card, border_radius=6)
        lbl_pos_title = font_bold.render("MACHINE PLACEMENT", True, COLOR_TEXT)
        screen.blit(lbl_pos_title, (25, 490))
        draw_card_help_button(screen, "machine", machine_card, font_small)
        source_label = DWELLING_SOURCE_MODES[dwelling_source_idx]
        scenario_short = current_scenario_label[-22:]
        lbl_scenario = font_small.render(f"Source: {source_label[:10]} / {scenario_short}", True, COLOR_TEXT)
        screen.blit(lbl_scenario, (25, 510))
        frame = current_scenario_summary.get("routing_frame") or {}
        frame_name = str(frame.get("name") or ROUTING_FRAME_OPTIONS[routing_frame_idx]).replace("_", " ")
        lbl_frame = font_small.render(f"Frame: {frame_name[:24]}", True, COLOR_MUTED)
        screen.blit(lbl_frame, (25, 530))
        lbl_coord = font_small.render(f"Position: ({int(machine_cx)}, {int(machine_cy)}) mm", True, COLOR_TEXT)
        screen.blit(lbl_coord, (25, 550))
        rot_mode_short = "Field" if rotation_mode_idx == 1 else "Torque"
        if rotation_mode_idx == 1:
            h_score = rotation_field_scores.get("H", 0.0)
            v_score = rotation_field_scores.get("V", 0.0)
            selected = rotation_field_scores.get("selected") or "-"
            rot_text = f"Rot: {machine_angle}° {rot_mode_short} {selected} H{h_score:.3f}/V{v_score:.3f}"
        else:
            rot_text = f"Rotation: {machine_angle}° / {rot_mode_short}"
        lbl_rot = font_small.render(rot_text[:38], True, COLOR_TEXT)
        screen.blit(lbl_rot, (25, 570))
        
        # 4. KPI Metrics Card
        kpi_card = pygame.Rect(15, 595, CANVAS_LEFT - 40, 135)
        pygame.draw.rect(screen, (40, 45, 55), kpi_card, border_radius=6)
        lbl_kpi_title = font_bold.render("ROUTING RUNTIME KPIs", True, COLOR_TEXT)
        screen.blit(lbl_kpi_title, (25, 605))
        draw_card_help_button(screen, "kpi", kpi_card, font_small)
        
        total_len_mm = 0.0
        total_turns_count = 0
        if routes:
            for name, segs in routes:
                total_len_mm += sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in segs)
            total_turns_count = count_solution_turns(routes)
                
        lbl_len = font_small.render(f"Total Duct Length: {total_len_mm/1000.0:.2f} m", True, COLOR_TEXT)
        screen.blit(lbl_len, (25, 625))
        lbl_turns = font_small.render(f"Total Turns: {total_turns_count}", True, COLOR_TEXT)
        screen.blit(lbl_turns, (25, 645))
        crossings_count = count_segment_crossings(routes) if routes else 0
        lbl_cross = font_small.render(f"Duct Crossings: {crossings_count}", True, COLOR_TEXT)
        screen.blit(lbl_cross, (25, 665))
        short_pieces_count = count_solution_short_pieces(routes) if routes else 0
        lbl_short = font_small.render(f"Short Pieces: {short_pieces_count}", True, COLOR_TEXT)
        screen.blit(lbl_short, (25, 685))
        lbl_score = font_small.render(f"Total Cost Score: {get_solution_score(routes, crossings_count) if routes else 0}", True, COLOR_TEXT)
        screen.blit(lbl_score, (25, 705))
        
        # 5. Status Box
        status_card = pygame.Rect(15, 740, CANVAS_LEFT - 40, 170)
        pygame.draw.rect(screen, (40, 45, 55), status_card, border_radius=6)
        lbl_status_title = font_bold.render("SOLVER EXECUTION STATUS", True, COLOR_TEXT)
        screen.blit(lbl_status_title, (25, 750))
        draw_card_help_button(screen, "status", status_card, font_small)
        
        words = status.split()
        lines = []
        curr_line = ""
        for w in words:
            if len(curr_line + " " + w) > 28:
                lines.append(curr_line)
                curr_line = w
            else:
                curr_line = (curr_line + " " + w).strip()
        if curr_line:
            lines.append(curr_line)
            
        y_off = 770
        for ln in lines:
            lbl_line = font_small.render(ln, True, COLOR_TEXT)
            screen.blit(lbl_line, (25, y_off))
            y_off += 18

        validation_warnings = get_route_validation_warnings(routes)
        warn_text = "Warnings: none" if not validation_warnings else "Warnings: " + ", ".join(validation_warnings[:2])
        if len(validation_warnings) > 2:
            warn_text += f" +{len(validation_warnings) - 2}"
        warn_color = COLOR_MUTED if not validation_warnings else (241, 196, 15)
        lbl_warn = font_small.render(warn_text[:42], True, warn_color)
        screen.blit(lbl_warn, (25, 835))
            
        lbl_runtime = font_small.render(f"Pathfinder time: {elapsed_ms:.1f} ms", True, COLOR_TEXT)
        screen.blit(lbl_runtime, (25, 865))
        lbl_nodes = font_small.render(f"Total routed nodes: {total_nodes}", True, COLOR_MUTED)
        screen.blit(lbl_nodes, (25, 885))
        lbl_fps = font_small.render(f"Render engine: Pygame ({clock.get_fps():.0f} FPS)", True, COLOR_MUTED)
        screen.blit(lbl_fps, (25, 905))
        
        # ── RIGHT PANEL: plots ──────────────────────────────────────────────
        draw_viewer_legend(screen, font_small)

        panel_x = WINDOW_WIDTH - PANEL_W
        pygame.draw.rect(screen, COLOR_PANEL, (panel_x, 0, PANEL_W, WINDOW_HEIGHT))
        pygame.draw.line(screen, (55, 55, 70), (panel_x, 0), (panel_x, WINDOW_HEIGHT))
        lbl_panel = font_bold.render("PLACEMENT EXPLORER", True, COLOR_MUTED)
        screen.blit(lbl_panel, (panel_x + PANEL_W // 2 - lbl_panel.get_width() // 2, 8))
        draw_plots(screen, font_small, font_bold)
        draw_solution_logs_panel(screen, font_small, font_bold)
        draw_help_popup(screen, font_small)
        draw_transient_message(screen, font_small)
        draw_terminal_validity_tooltip(screen, font_small)

        pygame.display.flip()
        clock.tick(FPS)
        
    pygame.quit()

if __name__ == "__main__":
    main()
