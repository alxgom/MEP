import sys
import os
import traceback
from pathlib import Path
import pygame
from mep_routing.domain import (
    local_axis_to_world as _local_axis_to_world,
)
from mep_routing.app import (
    build_demo_config,
    DemoRoutingController,
    DemoWorkbenchApplication,
    DwellingRuntime,
    InstallationGraphProfile,
    InstallationMachineUi,
    InstallationUiSession,
    run_workbench_smoke,
    WorkbenchState,
)
from mep_routing.installations.cli import CLI_INSTALLATION, CliInteractiveRuntime, cli_machine_pins
from mep_routing.installations.coc import COC_INSTALLATION, CocInteractiveRuntime
from mep_routing.data_sources import (
    discover_dwelling_cases as _discover_dwelling_cases,
    build_core_shaft_entry_specs as _source_core_shaft_entry_specs,
)
from mep_routing.installations.sal import (
    SAL_INSTALLATION,
)
from mep_routing.installations.sal.graph_inputs import build_sal_simple_graph_inputs
from mep_routing.geometry import iter_polygons as _iter_polygons
from mep_routing.placement import (
    PlacementSession,
)
from mep_routing.routing import (
    count_segment_crossings as _count_segment_crossings,
    count_solution_turns as _count_solution_turns,
    total_route_length_m as _total_route_length_m,
)
from mep_routing.ui.runtime import CanvasUiRuntime
from mep_routing.ui import (
    DemoUiFacade,
    WorkbenchChromeRuntime,
    WorkbenchCommandController,
    WorkbenchTheme,
)
from mep_routing.ui.installation_projection import active_capabilities
from mep_routing.ui.terminal_runtime import TerminalUiRuntime, TerminalUiStyle
from mep_routing.ui.drawing import draw_outlined_text as _draw_outlined_text
from mep_routing.ui.overlays import (
    draw_terminal_area_drag as _draw_terminal_area_drag,
    draw_wet_room_outer_accents as _draw_wet_room_outer_accents,
)
from mep_routing.observability import InteractiveObservationController
from mep_routing.ui.installation_header import (
    InstallationHeaderState,
    InstallationOption,
)
from mep_routing.ui.workbench_loop import run_pygame_workbench

# Add relative paths to sys.path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '08-bend-aware-non-orthogonal')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

demo_config = build_demo_config(SAL_INSTALLATION)
theme = WorkbenchTheme()
if demo_config.dwelling_export_root.exists():
    sys.path.append(str(demo_config.dwelling_export_root))

import generative_layout
from solver import estimate_turns

try:
    from dwelling_export.demo_loader import load_dwelling_scenario, scenario_summary
except ImportError:
    load_dwelling_scenario = None
    scenario_summary = None

# Live solver choices; immutable defaults and labels live in ``demo_config``.
C_BEND = demo_config.bend_default
crossing_penalty_multiplier = demo_config.crossing_default
min_piece_factor = demo_config.min_piece_factor_default

graph_type_idx = demo_config.graph_types.index(SAL_INSTALLATION.default_graph_mode)

routing_strategy_idx = SAL_INSTALLATION.routing_strategies.index(
    SAL_INSTALLATION.default_routing_strategy
)

router_backend_idx = SAL_INSTALLATION.search_backends.index(
    SAL_INSTALLATION.default_search_backend
)

heuristic_mode_idx = demo_config.heuristic_modes.index(SAL_INSTALLATION.default_heuristic_mode)

auto_placement_mode_idx = 2
rotation_mode_idx = 0
rotation_field_scores = {"H": 0.0, "V": 0.0, "selected": None}
show_grid_graph = False
show_heatmap = False
edge_weight_heatmap_enabled = False
edge_weight_view_mode_idx = 0
route_real_diameter_width_enabled = False
show_cli_core_polygons = False
preferred_terminal_tool_mode = None
terminal_validity_overlay_enabled = False
room_start_mode_idx = 0
dwelling_source_idx = 0

INSTALLATION_OPTIONS = (
    InstallationOption("sal", "Sal", available=True),
    InstallationOption("cli", "Cli", available=True),
    InstallationOption("san", "San"),
    InstallationOption("coc", "Coc", available=True),
)
installation_header_state = InstallationHeaderState(
    order=tuple(option.key for option in INSTALLATION_OPTIONS),
    enabled=frozenset({"sal"}),
    active="sal",
)
installation_ui_session = InstallationUiSession.create({
    "sal": InstallationMachineUi(
        SAL_INSTALLATION.machine_catalog, runtime_available=True,
    ),
    "cli": InstallationMachineUi(
        CLI_INSTALLATION.machine_catalog,
        runtime_available=CLI_INSTALLATION.interactive_runtime_available,
    ),
    "coc": InstallationMachineUi(COC_INSTALLATION.source_catalog, runtime_available=True),
})
workbench_state = WorkbenchState.create(("sal", "cli", "coc"))
cli_runtime = CliInteractiveRuntime(
    CLI_INSTALLATION, demo_config.common_graph, demo_config.terminal_runtime,
)
coc_runtime = CocInteractiveRuntime(
    COC_INSTALLATION, demo_config.common_graph, demo_config.terminal_runtime,
)

canvas_ui = CanvasUiRuntime()
is_fullscreen = False
preference_reset_rect = pygame.Rect(0, 0, 1, 1)
terminal_ui = TerminalUiRuntime(canvas_ui, TerminalUiStyle(
    route_colors=theme.route_colors, text_color=theme.text,
    allowed_color=theme.terminal_allowed, blocked_color=theme.terminal_blocked,
    blocked_hatch_color=theme.terminal_blocked_hatch,
    marker_size_mm=demo_config.preferred_terminal_marker_size_mm,
    remap_tolerance_mm=demo_config.preferred_terminal_remap_tolerance_mm,
    regulation_clearance_mm=demo_config.terminal_clearance_mm,
    room_buffer_mm=demo_config.terminal_room_buffer_mm,
))
workbench_chrome = WorkbenchChromeRuntime(
    canvas_ui.viewport, terminal_ui, theme.panel_width, theme.chrome_colors(),
)

try:
    REAL_DWELLING_SCENARIOS = _discover_dwelling_cases(demo_config.real_dwelling_db)
except (ImportError, OSError) as error:
    print(f"Real dwelling catalog unavailable: {error}")
    REAL_DWELLING_SCENARIOS = ()
if not REAL_DWELLING_SCENARIOS:
    dwelling_source_idx = demo_config.dwelling_source_modes.index("Random Synthetic")
real_scenario_idx = 0
routing_frame_idx = 0
current_scenario_label = "synthetic"
current_scenario_summary = {}

# Observation state for the right-side plots and solution-log panel.
weight_mode_idx = 0                             # 0 for Default, 1 for Equal Weights
heatmap_scale_mode = 0                          # 0 for Linear (75% Saturation), 1 for Log Scale
heatmap_palette_idx = 0                         # 0 for Turbo, 1 for Viridis
to_screen = canvas_ui.viewport.to_screen
to_mm = canvas_ui.viewport.to_world


application = DemoWorkbenchApplication(sys.modules[__name__])
workbench_commands = WorkbenchCommandController(application, pygame)
ui_facade = DemoUiFacade(application, pygame)

ap_scores = {}
ap_fields = {}
routing_controller = DemoRoutingController(application)
dwelling_runtime = DwellingRuntime(application)
application.commands = workbench_commands
application.ui_facade = ui_facade
application.routing_controller = routing_controller
application.dwelling_runtime = dwelling_runtime
# ROUTING UTILITIES AND CONSTRAINTS
SAL_GRAPH_PROFILE = InstallationGraphProfile(
    routing_controller.port_access_specs, build_sal_simple_graph_inputs,
    SAL_INSTALLATION.default_machine.small_duct_diameter_mm,
)

# TOPOLOGICAL DISTANCE FIELDS AUTO-PLACEMENT ALGORITHMS
placement_session = PlacementSession(
    machine_area=(
        SAL_INSTALLATION.default_machine.overall_width_mm
        * SAL_INSTALLATION.default_machine.body_height_mm
    ),
    machine_pins=routing_controller.machine_pins,
    representative_point=application.get_representative_point,
    route_room_polygon=terminal_ui.route_room_polygon,
    local_axis_to_world=_local_axis_to_world,
    graph_builder=routing_controller.build_grid,
    shaft_nodes=routing_controller.shaft_entry_nodes,
    weight_mode=weight_mode_idx,
    rotation_eps=demo_config.rotation_field_epsilon,
)



observation_controller = InteractiveObservationController(application, workbench_chrome)
application.observation = observation_controller
application.chrome = workbench_chrome


def main():
    run_pygame_workbench(
        application,
        caption="Integrated MEP Routing Workbench (Demo 10.8.1)",
        fps=demo_config.fps,
    )


def smoke_test():
    """Exercise application startup and routing without opening a window."""

    return run_workbench_smoke(application)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        error_log = Path(__file__).with_name("runtime_stderr.log")
        with error_log.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        raise
