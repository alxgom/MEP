"""Pygame bootstrap and frame loop for the routing workbench."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from .canvas import CanvasFonts, CanvasRenderHooks, draw_canvas_scene
from .frame import (
    CanvasFrameCallbacks, CanvasFramePalette, CanvasFrameState, SidebarFrameState,
    active_selected_route, build_canvas_scene, build_sidebar_view,
)
from .installation_header import draw_installation_header
from .installation_projection import InstallationVisual, installation_context_layers
from .pygame_events import WorkbenchEventAdapter
from .sidebar import SidebarColors, SidebarFonts, draw_sidebar


@dataclass(frozen=True)
class WorkbenchFonts:
    title: object
    bold: object
    small: object
    micro: object
    plot_small: object
    plot_title: object
    plot_value: object
    plot_minimum: object


def _fonts():
    family = "Arial, Liberation Sans, DejaVu Sans"
    return WorkbenchFonts(
        pygame.font.SysFont("Outfit", 24, bold=True),
        pygame.font.SysFont("Outfit", 18, bold=True),
        pygame.font.SysFont("Outfit", 15),
        pygame.font.SysFont("Outfit", 11, bold=True),
        pygame.font.SysFont(family, 11),
        pygame.font.SysFont(family, 16, bold=True),
        pygame.font.SysFont(family, 14, bold=True),
        pygame.font.SysFont(family, 13),
    )


def _render(app, screen, clock, fonts, state):
    routes, status = state.routes, state.status
    selected = active_selected_route(state.selected_route_name, routes)
    state.selected_route_name = selected
    selected_room = app.get_route_room_polygon(selected) if selected else None
    active = app.installation_header_state.active
    workbench = app.workbench_state
    dwelling_state = workbench.dwelling
    sal_pose = workbench.installation("sal").pose
    cli_pose = workbench.installation("cli").pose
    cli_result = app.cli_runtime.last_result
    workspace = cli_result.workspace if active == "cli" and cli_result.workspace is not None else dwelling_state.workspace
    terminals = cli_result.terminals if active == "cli" else app.terminals
    pose = app.get_active_machine_pose()
    pins = app.get_active_machine_pins(*pose)
    selected_pins = (
        ({"air_out"} if selected == "Supply Air Tree" else set())
        if active == "cli" else (app.get_selected_pin_names(selected, routes, pins) if selected else set())
    )
    machine_key = app.installation_ui_session.selected_keys["cli"]
    context = installation_context_layers(
        app.installation_header_state.order, app.installation_header_state.enabled,
        active, workbench.runs,
        {
            "sal": InstallationVisual(app.terminals, app.get_machine_pins(*sal_pose.as_tuple())),
            "cli": InstallationVisual(
                cli_result.terminals,
                app.cli_machine_pins(app.CLI_INSTALLATION.machine_catalog.value(machine_key),
                                     cli_pose.center, cli_pose.angle_degrees),
            ),
        },
    )
    if app.show_heatmap:
        app.ensure_placement_heatmap_scores()

    def edge_overlay():
        if app.edge_weight_heatmap_enabled:
            app.refresh_edge_weight_view_overlay(routes)
        app.draw_edge_weight_heatmap(screen)

    scene = build_canvas_scene(CanvasFrameState(
        rooms=app.rooms, doors=app.doors, columns=app.columns, shafts=app.shafts,
        shaft_extraction=app.shaft_extraction, graph_env=workspace.env,
        show_grid_graph=app.show_grid_graph, terminals=terminals, routes=routes or (),
        selected_route_name=selected, selected_room_polygon=selected_room,
        global_pins=pins, selected_pins=selected_pins, context_layers=context,
        auto_placement_mode=(app.auto_placement_mode_idx if active == "sal" else 0),
        placement_fields=(app.ap_fields if active == "sal" else {}),
        wet_room_names=app.wet_room_names, route_colors=app.ROUTE_COLORS,
        palette=CanvasFramePalette(
            app.COLOR_BG, app.COLOR_WALL, app.COLOR_DOOR, app.COLOR_COLUMN, app.COLOR_SHAFT,
            app.COLOR_SHAFT_INACTIVE, app.COLOR_GRAPH_EDGE, app.COLOR_GRAPH_NODE, app.COLOR_ROOM,
            app.COLOR_ROOM_COVERED, app.COLOR_DESELECTED_ROOM, app.COLOR_DESELECTED_PIN,
            app.COLOR_DESELECTED_ROUTE, app.COLOR_PLAN_LABEL, app.COLOR_TEXT, app.COLOR_SELECTION_HALO,
        ),
        callbacks=CanvasFrameCallbacks(app.to_screen, app.get_representative_point,
                                       app.get_route_draw_width, workspace.spatial_index.query),
        room_wall_width=app.WALL_DRAW_WIDTH,
        terminal_area_start=state.terminal_area_start_mm,
        terminal_area_end=state.terminal_area_end_mm if state.terminal_area_dragging else None,
    ))
    hooks = CanvasRenderHooks(
        draw_covers=lambda: app.canvas_ui.draw_geometry(screen, app.covers, app.COLOR_COVER_OVERLAY),
        draw_distance_heatmap=lambda: app.draw_distance_heatmap(screen, app.ap_scores) if app.show_heatmap and app.ap_scores else None,
        draw_distance_colorbar=lambda: app.draw_colorbar(screen, app.get_placeable_heatmap_scores(app.ap_scores)) if app.show_heatmap and app.ap_scores and not app.edge_weight_heatmap_enabled else None,
        draw_edge_weight_heatmap=edge_overlay,
        draw_edge_weight_colorbar=lambda: app.draw_edge_weight_colorbar(screen),
        draw_terminal_validity_overlay=lambda: app.terminal_ui.draw_validity_overlay(screen, app.terminal_validity_overlay_enabled and active == "sal"),
        draw_wet_room_accents=lambda: app.draw_wet_room_outer_accents(screen),
        draw_polygon_hatch=lambda shaft: app.canvas_ui.draw_hatch(screen, shaft, app.COLOR_SHAFT_INACTIVE_HATCH, spacing=9),
        draw_outlined_text=lambda target, font, label, position, color: app.draw_outlined_text(target, font, label, position, color),
        draw_preferred_terminal_areas=lambda value: app.terminal_ui.draw_preferred_areas(screen, value) if active == "sal" else None,
        draw_routed_terminal_endpoints=lambda value: app.terminal_ui.draw_routed_endpoints(screen, routes, value) if active == "sal" else None,
        draw_preferred_terminal_markers=lambda value: app.terminal_ui.draw_preferred_markers(screen, value, routes) if active == "sal" else None,
        draw_terminal_area_drag=lambda start, end: app.draw_terminal_area_drag(screen, start, end),
    )
    draw_canvas_scene(screen, scene=scene, fonts=CanvasFonts(small=fonts.small), hooks=hooks)
    draw_installation_header(
        screen, fonts.small, fonts.micro, app.installation_header_state,
        app.INSTALLATION_OPTIONS, app.get_installation_header_layout(),
        dragged_key=state.installation_drag_key,
    )
    app.draw_ruler_overlay(screen, fonts.small, state.ruler_start_mm, state.ruler_end_mm)
    app.draw_canvas_tool_controls(screen, fonts.small, state.ruler_mode, state.dwelling_selector_open)
    if active == "sal":
        app.draw_terminal_tool_buttons(screen, fonts.bold, fonts.small)

    terminal_runtime = dwelling_state.terminal_runtime
    preferred_count = sum(len(points) for points in (() if terminal_runtime is None else terminal_runtime.preferred_points_by_room.values()))
    warnings = app.get_route_validation_warnings(routes) if active == "sal" else []
    options = app.installation_ui_session.options(active)
    dropdown = app.installation_ui_session.dropdown_state(active)
    available = app.installation_ui_session.installations[active].runtime_available
    sidebar = build_sidebar_view(SidebarFrameState(
        auto_placement_mode=(app.AUTO_PLACEMENT_MODES[app.auto_placement_mode_idx] if active == "sal" else "Cli placement pending"),
        show_heatmap=app.show_heatmap, heatmap_scale_mode=app.heatmap_scale_mode,
        heatmap_palette_index=app.heatmap_palette_idx, weight_mode_index=app.weight_mode_idx,
        rotation_mode_index=app.rotation_mode_idx, rotation_field_scores=app.rotation_field_scores,
        machine_angle=pose[2], strategy=(app.ROUTING_STRATEGIES[app.routing_strategy_idx] if active == "sal" else "Kou Steiner tree"),
        router=(app.ROUTER_BACKENDS[app.router_backend_idx] if active == "sal" else "Cli supply runtime"),
        heuristic=(app.HEURISTIC_MODES[app.heuristic_mode_idx] if active == "sal" else "Directional scales"),
        graph_type=(app.GRAPH_TYPES[app.graph_type_idx] if active == "sal" else "Core Simple"),
        room_start_mode=(app.ROOM_START_MODES[app.room_start_mode_idx] if active == "sal" else "Supply terminals"),
        selected_route_name=selected, preferred_terminal_count=preferred_count,
        bend_value=app.C_BEND, bend_min=app.C_BEND_MIN, bend_max=app.C_BEND_MAX,
        crossing_value=app.crossing_penalty_multiplier, crossing_min=app.CROSSING_MULTIPLIER_MIN,
        crossing_max=app.CROSSING_MULTIPLIER_MAX, scenario_summary=app.current_scenario_summary,
        fallback_frame_name=app.ROUTING_FRAME_OPTIONS[app.routing_frame_idx], machine_position=pose[:2],
        status=status, validation_warnings=warnings, elapsed_ms=state.elapsed_ms,
        total_nodes=state.total_nodes, fps=clock.get_fps(), machine_options=options,
        selected_machine_key=dropdown.selected_key, machine_selector_open=dropdown.open,
        machine_runtime_available=available,
    ))
    viewport = app.canvas_ui.viewport
    draw_sidebar(
        screen, canvas_left=viewport.canvas_left, window_height=viewport.window_height,
        fonts=SidebarFonts(title=fonts.title, bold=fonts.bold, small=fonts.small),
        colors=SidebarColors(panel=app.COLOR_PANEL, wall=app.COLOR_WALL, text=app.COLOR_TEXT, muted=app.COLOR_MUTED),
        view=sidebar, draw_help_button=lambda card, rect, font: app.draw_card_help_button(screen, card, rect, font),
        draw_min_piece_slider=app.draw_min_piece_slider, draw_weight_slider=app.draw_weight_slider,
    )
    app.draw_viewer_legend(screen, fonts.small)
    panel_x = viewport.window_width - app.PANEL_W
    pygame.draw.rect(screen, app.COLOR_PANEL, (panel_x, 0, app.PANEL_W, viewport.window_height))
    pygame.draw.line(screen, (55, 55, 70), (panel_x, 0), (panel_x, viewport.window_height))
    label = fonts.bold.render("PLACEMENT EXPLORER", True, app.COLOR_MUTED)
    screen.blit(label, (panel_x + app.PANEL_W // 2 - label.get_width() // 2, 8))
    app.draw_plots(screen, fonts.plot_small, fonts.plot_title, fonts.plot_value, fonts.plot_minimum)
    app.draw_solution_logs_panel(screen, fonts.small, fonts.bold)
    app.draw_help_popup(screen, fonts.small)
    app.draw_transient_message(screen, fonts.small)
    if active == "sal":
        app.terminal_ui.draw_validity_tooltip(screen, fonts.small, pygame.mouse.get_pos(), app.terminal_validity_overlay_enabled)


def run_pygame_workbench(app, *, caption, fps):
    """Initialize Pygame and run the workbench until its event state stops."""
    pygame.init()
    pygame.font.init()
    viewport = app.canvas_ui.viewport
    app.canvas_ui.update_layout(viewport.window_width, viewport.window_height)
    screen = pygame.display.set_mode(viewport.window_size, pygame.RESIZABLE)
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()
    fonts = _fonts()
    app.generate_new_dwelling()
    events = WorkbenchEventAdapter(app, screen, app.solve_ventilation_routing())
    try:
        while events.state.running:
            events.run_pending_auto_placement()
            state = events.process()
            screen = events.screen
            _render(app, screen, clock, fonts, state)
            pygame.display.flip()
            clock.tick(fps)
    finally:
        pygame.quit()
