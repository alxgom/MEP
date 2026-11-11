"""Pygame bootstrap and frame loop for the routing workbench."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from mep_routing.installations.cli.placement_mask import cell_center_product_candidates

from .canvas import CanvasFonts, CanvasRenderHooks, draw_canvas_scene
from .frame import (
    CanvasFrameCallbacks, CanvasFramePalette, CanvasFrameState, SidebarFrameState,
    active_selected_route, build_canvas_scene, build_sidebar_view,
)
from .installation_header import draw_installation_header
from .installation_projection import project_workbench
from .heatmaps import draw_placement_cell_grid, draw_placement_mask, draw_placement_mask_legend
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
    selected_room = app.terminal_ui.route_room_polygon(selected) if selected else None
    active = app.installation_header_state.active
    workbench = app.workbench_state
    dwelling_state = workbench.dwelling
    projection = project_workbench(app)
    active_view = projection.active
    workspace, terminals = active_view.workspace, active_view.terminals
    pose, pins = active_view.pose, active_view.pins
    selected_pins = (
        app.routing_controller.selected_pin_names(selected, routes, pins)
        if selected and active_view.can("terminal_tools")
        else ({"air_out"} if selected == "Supply Air Tree" else set())
    )
    context = projection.context_layers
    sal_heatmap = app.show_heatmap and active_view.can("placement_overlay")
    cli_heatmap = active == "cli" and workbench.cli.show_placement_mask
    virtual_placement = (active == "cli"
                         and workbench.cli.auto_placement_mode_idx == app.CLI_INSTALLATION.virtual_placement_mode_index)
    viewport = app.canvas_ui.viewport
    placement_mask = None
    placement_outlines = ()
    placement_cell_grid = None
    placement_candidates = ()
    placement_cell_label = None
    if cli_heatmap and active_view.can("machine_selection") and dwelling_state.active is not None:
        runtime = app.routing_controller.installations.get("cli")
        placement_mask = runtime.placement_mask(dwelling_state.active.prepared, pose[2])
        if placement_mask is not None:
            placement_outlines = runtime.placement_outlines(dwelling_state.active.prepared, pins, pose[2])
            if workbench.cli.show_placement_cells and virtual_placement:
                search = app.cli_runtime.virtual_placement
                placement_cell_grid = search.candidate_grid(dwelling_state.active.prepared)
                placement_candidates = search.screened_centers(dwelling_state.active.prepared, placement_mask)
                placement_cell_label = f"Hanan + Cell · {len(placement_candidates)} candidates"
            elif workbench.cli.show_placement_cells:
                room_aware = workbench.cli.room_aware_placement_cells
                placement_cell_grid = runtime.placement_cell_grid(dwelling_state.active, room_aware=room_aware)
                if placement_cell_grid is not None:
                    placement_candidates = cell_center_product_candidates(
                        placement_mask, placement_cell_grid.diagnostics["cells"],
                    )
                    mode = "Room-aware" if room_aware else "Original cells"
                    placement_cell_label = f"{mode} · {len(placement_candidates)} candidates"
    if sal_heatmap:
        app.routing_controller.ensure_placement_scores()

    def placement_overlay():
        if placement_mask is not None:
            draw_placement_mask(screen, placement_mask, viewport)
            if placement_cell_grid is not None:
                draw_placement_cell_grid(screen, placement_cell_grid, placement_candidates, viewport)
        elif sal_heatmap and app.ap_scores:
            app.ui_facade.draw_distance_heatmap(screen, app.ap_scores)

    def placement_legend():
        if app.edge_weight_heatmap_enabled:
            return
        if placement_mask is not None:
            draw_placement_mask_legend(screen, placement_mask, viewport, fonts.small, pose[2],
                                       placement_cell_label, compare_cells=not virtual_placement)
        elif sal_heatmap and app.ap_scores:
            app.ui_facade.draw_colorbar(screen, app.ui_facade.get_placeable_heatmap_scores(app.ap_scores))

    def edge_overlay():
        if app.edge_weight_heatmap_enabled:
            if active_view.can("routing_weights"):
                app.routing_controller.refresh_edge_weight_overlay(routes, workspace)
            else:
                workspace.overlay.reset()
        app.ui_facade.draw_edge_weight_heatmap(screen, workspace)

    scene = build_canvas_scene(CanvasFrameState(
        rooms=app.rooms, doors=app.doors, columns=app.columns, shafts=app.shafts,
        shaft_extraction=getattr(active_view.result, "shaft_extraction", app.shaft_extraction),
        graph_env=workspace.env,
        show_grid_graph=app.show_grid_graph, terminals=terminals,
        terminal_segments=active_view.terminal_segments,
        terminal_arrows=active_view.terminal_arrows, routes=routes or (),
        selected_route_name=selected, selected_room_polygon=selected_room,
        global_pins=pins, selected_pins=selected_pins, context_layers=context,
        auto_placement_mode=(app.auto_placement_mode_idx if active_view.can("placement_overlay") else 0),
        placement_fields=(app.ap_fields if active_view.can("placement_overlay") else {}),
        wet_room_names=app.wet_room_names, route_colors=app.theme.route_colors,
        palette=CanvasFramePalette(
            app.theme.background, app.theme.wall, app.theme.door, app.theme.column,
            app.theme.shaft, app.theme.inactive_shaft, app.theme.graph_edge,
            app.theme.graph_node, app.theme.room, app.theme.covered_room,
            app.theme.deselected_room, app.theme.deselected_pin,
            app.theme.deselected_route, app.theme.plan_label, app.theme.text,
            app.theme.selection_halo,
        ),
        callbacks=CanvasFrameCallbacks(app.to_screen, app.get_representative_point,
                                       app.ui_facade.get_route_draw_width,
                                       workspace.spatial_index.query),
        room_wall_width=app.theme.wall_draw_width,
        terminal_area_start=state.terminal_area_start_mm,
        terminal_area_end=state.terminal_area_end_mm if state.terminal_area_dragging else None,
        placement_outlines=placement_outlines,
        placement_center=pose[:2] if placement_mask is not None else None,
    ))
    hooks = CanvasRenderHooks(
        draw_covers=lambda: app.canvas_ui.draw_geometry(screen, app.covers, app.theme.cover_overlay),
        draw_distance_heatmap=placement_overlay,
        draw_distance_colorbar=placement_legend,
        draw_edge_weight_heatmap=edge_overlay,
        draw_edge_weight_colorbar=lambda: app.ui_facade.draw_edge_weight_colorbar(screen, workspace),
        draw_route_polygons=lambda: app.canvas_ui.draw_geometry(
            screen,
            (
                app.cli_runtime.core_polygon_projection(active_view.result).polygons
                if app.workbench_state.cli.duct_geometry_mode_idx == 1
                else app.cli_runtime.aligned_polygon_projection(active_view.result).polygons
            ),
            (*app.theme.installation_colors["cli"], 86),
        ) if active == "cli" and app.workbench_state.cli.duct_geometry_mode_idx and active_view.result is not None else None,
        draw_terminal_validity_overlay=lambda: app.terminal_ui.draw_validity_overlay(screen, app.terminal_validity_overlay_enabled and active_view.can("terminal_tools")),
        draw_wet_room_accents=lambda: app.draw_wet_room_outer_accents(screen),
        draw_polygon_hatch=lambda shaft: app.canvas_ui.draw_hatch(
            screen, shaft, app.theme.inactive_shaft_hatch, spacing=9,
        ),
        draw_outlined_text=lambda target, font, label, position, color: app.draw_outlined_text(target, font, label, position, color),
        draw_preferred_terminal_areas=lambda value: app.terminal_ui.draw_preferred_areas(screen, value) if active_view.can("terminal_tools") else None,
        draw_routed_terminal_endpoints=lambda value: app.terminal_ui.draw_routed_endpoints(screen, routes, value) if active_view.can("terminal_tools") else None,
        draw_preferred_terminal_markers=lambda value: app.terminal_ui.draw_preferred_markers(screen, value, routes) if active_view.can("terminal_tools") else None,
        draw_terminal_area_drag=lambda start, end: app.draw_terminal_area_drag(screen, start, end),
    )
    draw_canvas_scene(screen, scene=scene, fonts=CanvasFonts(small=fonts.small), hooks=hooks)
    draw_installation_header(
        screen, fonts.small, fonts.micro, app.installation_header_state,
        app.INSTALLATION_OPTIONS, app.ui_facade.installation_header_layout(),
        dragged_key=state.installation_drag_key if state.installation_drag_enabled else None,
        dragged_position=state.installation_drag_position,
    )
    app.ui_facade.draw_ruler_overlay(
        screen, fonts.small, state.ruler_start_mm, state.ruler_end_mm,
    )
    app.ui_facade.draw_canvas_tool_controls(
        screen, fonts.small, state.ruler_mode, state.dwelling_selector_open,
    )
    if active_view.can("terminal_tools"):
        app.ui_facade.draw_terminal_tool_buttons(screen, fonts.bold, fonts.small)

    terminal_runtime = dwelling_state.terminal_runtime
    preferred_count = sum(len(points) for points in (() if terminal_runtime is None else terminal_runtime.preferred_points_by_room.values()))
    warnings = app.routing_controller.validation_warnings(routes) if active_view.can("route_validation") else []
    options = app.installation_ui_session.options(active)
    dropdown = app.installation_ui_session.dropdown_state(active)
    solver_dropdowns = app.ui_facade.solver_dropdowns()
    available = app.installation_ui_session.installations[active].runtime_available
    cli_state = app.workbench_state.cli
    solver_detail = None
    placement_weights_label = None
    if active == "cli":
        auto_placement_mode = app.CLI_INSTALLATION.auto_placement_labels[
            cli_state.auto_placement_mode_idx
        ]
        bend_value = cli_state.bend_penalty_mm
        crossing_value = cli_state.crossing_penalty_multiplier
        placement_weights_label = "Grille-distance ranking"
        if virtual_placement:
            placement_weights_label = "Shared supply-tree cost"
            placement = app.cli_runtime.last_result.placement
            if (placement is not None and (*placement.center, placement.angle_degrees) == pose
                    and placement.green_score_gap_mm is not None):
                gap = round(placement.green_score_gap_mm - cli_state.yellow_placement_penalty_mm)
                if gap == 0:
                    placement_weights_label = "Green and yellow scores tie"
                else:
                    leader = "Yellow" if gap > 0 else "Green"
                    placement_weights_label = f"{leader} leads by {abs(gap):,} mm"
        if cli_state.router_backend_idx in app.CLI_INSTALLATION.length_backend_indices:
            stats = app.cli_runtime.last_result.supply_length_stats
            if stats is not None and stats.fittings is not None:
                fits = stats.fittings
                solver_detail = f"Fittings: {fits.total}  (E {fits.elbows}, T {fits.tees}, X {fits.crosses})"
    elif active == "coc":
        auto_placement_mode = "Fixed exported source"
        bend_value = 0.0
        crossing_value = 0.0
        placement_weights_label = "Not applicable"
    else:
        auto_placement_mode = app.demo_config.auto_placement_modes[app.auto_placement_mode_idx]
        bend_value = app.C_BEND
        crossing_value = app.crossing_penalty_multiplier
    sidebar = build_sidebar_view(SidebarFrameState(
        auto_placement_mode=auto_placement_mode,
        show_heatmap=app.show_heatmap, heatmap_scale_mode=app.heatmap_scale_mode,
        heatmap_palette_index=app.heatmap_palette_idx, weight_mode_index=app.weight_mode_idx,
        rotation_mode_index=app.rotation_mode_idx, rotation_field_scores=app.rotation_field_scores,
        machine_angle=pose[2],
        strategy=solver_dropdowns[0].options[int(solver_dropdowns[0].selected_key)].label,
        router=solver_dropdowns[1].options[int(solver_dropdowns[1].selected_key)].label,
        heuristic=solver_dropdowns[2].options[int(solver_dropdowns[2].selected_key)].label,
        graph_type=solver_dropdowns[3].options[int(solver_dropdowns[3].selected_key)].label,
        room_start_mode=solver_dropdowns[4].options[int(solver_dropdowns[4].selected_key)].label,
        selected_route_name=selected, preferred_terminal_count=preferred_count,
        bend_value=bend_value, bend_min=app.demo_config.bend_min,
        bend_max=app.demo_config.bend_max,
        crossing_value=crossing_value,
        crossing_min=app.demo_config.crossing_min,
        crossing_max=app.demo_config.crossing_max,
        scenario_summary=app.current_scenario_summary,
        fallback_frame_name=app.demo_config.routing_frame_options[app.routing_frame_idx],
        machine_position=pose[:2],
        status=status, validation_warnings=warnings, elapsed_ms=state.elapsed_ms,
        total_nodes=state.total_nodes, fps=clock.get_fps(), machine_options=options,
        selected_machine_key=dropdown.selected_key, machine_selector_open=dropdown.open,
        machine_runtime_available=available,
        cli_return_ventilation_enabled=(
            app.workbench_state.cli.return_ventilation_enabled if active == "cli" else None
        ),
        solver_dropdowns=solver_dropdowns,
        solver_detail=solver_detail,
        placement_heatmap_label=(
            ("Body + supply turn [V]" if placement_mask is not None else "Unavailable [V]" if cli_heatmap else "Off · [V] Placement")
            if active == "cli" else "Not available" if active == "coc" else None
        ),
        placement_weights_label=placement_weights_label,
        placement_rotation_label=("Current rotation" if virtual_placement else
                                  "Machine-aligned" if active == "cli" else
                                  "Exported geometry" if active == "coc" else None),
        placement_shortcuts_label=("[A] Place | [P] Mode | [V] Placement" if active == "cli" else "Fixed source; no placement controls" if active == "coc" else None),
        yellow_placement_penalty=(
            (cli_state.yellow_placement_penalty_mm, app.CLI_INSTALLATION.yellow_placement_penalty_max_mm)
            if virtual_placement else None
        ),
    ))
    if not virtual_placement:
        app.canvas_ui.controls.yellow_placement_penalty = pygame.Rect(0, 0, 0, 0)
        app.canvas_ui.controls.yellow_placement_penalty_reset = pygame.Rect(0, 0, 0, 0)
    draw_sidebar(
        screen, canvas_left=viewport.canvas_left, window_height=viewport.window_height,
        fonts=SidebarFonts(title=fonts.title, bold=fonts.bold, small=fonts.small),
        colors=SidebarColors(
            panel=app.theme.panel, wall=app.theme.wall,
            text=app.theme.text, muted=app.theme.muted,
        ),
        view=sidebar,
        draw_help_button=lambda card, rect, font: app.observation.draw_card_help_button(
            screen, card, rect, font,
        ),
        draw_min_piece_slider=app.ui_facade.draw_min_piece_slider,
        draw_weight_slider=app.ui_facade.draw_weight_slider,
    )
    app.observation.draw_viewer_legend(screen, fonts.small)
    panel_x = viewport.window_width - app.theme.panel_width
    pygame.draw.rect(
        screen, app.theme.panel,
        (panel_x, 0, app.theme.panel_width, viewport.window_height),
    )
    pygame.draw.line(screen, (55, 55, 70), (panel_x, 0), (panel_x, viewport.window_height))
    label = fonts.bold.render("PLACEMENT EXPLORER", True, app.theme.muted)
    screen.blit(label, (panel_x + app.theme.panel_width // 2 - label.get_width() // 2, 8))
    app.observation.draw_plots(
        screen, fonts.plot_small, fonts.plot_title, fonts.plot_value, fonts.plot_minimum,
    )
    app.observation.draw_solution_logs_panel(screen, fonts.small, fonts.bold)
    app.observation.draw_help_popup(screen, fonts.small)
    app.observation.draw_transient_message(screen, fonts.small)
    if active_view.can("terminal_tools"):
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
    app.dwelling_runtime.generate()
    events = WorkbenchEventAdapter(
        app, screen, app.routing_controller.solve_installations(),
    )
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
