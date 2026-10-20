"""Thin UI facade for Demo 10.8.1's remaining callback surface."""

from __future__ import annotations

from .dropdown import DropdownOption, dropdown_bounds
from .installation_header import installation_header_layout
from .installation_projection import active_capabilities
from .sidebar import (
    CLI_RETURN_TOGGLE_BOUNDS,
    MACHINE_SELECTOR_Y,
    SolverDropdown,
    solver_dropdown_bounds,
)


SOLVER_CONTROL_FIELDS = (
    ("strategy", "Strategy", "routing_strategy_idx"),
    ("router", "Router", "router_backend_idx"),
    ("heuristic", "Heuristic", "heuristic_mode_idx"),
    ("graph", "Grid", "graph_type_idx"),
    ("starts", "Starts", "room_start_mode_idx"),
)


class DemoUiFacade:
    def __init__(self, app, pygame):
        self.app, self.pygame = app, pygame

    def installation_header_layout(self):
        a = self.app
        count = len(a.installation_header_state.order)
        width = max(420, a.canvas_ui.viewport.canvas_width - 28)
        pill = min(126, max(92, (width - (count - 1) * 30) // count))
        gap = min(42, max(24, (width - count * pill) // max(1, count - 1)))
        return installation_header_layout(
            a.canvas_ui.viewport.canvas_left, a.theme.installation_header_top,
            a.installation_header_state.order, pill_width=pill, gap=gap,
        )

    def handle_cli_return_toggle_click(self, pos):
        a = self.app
        if a.installation_header_state.active != "cli":
            return False
        if not self.pygame.Rect(CLI_RETURN_TOGGLE_BOUNDS).collidepoint(pos):
            return False
        state = a.workbench_state.cli
        state.return_ventilation_enabled = not state.return_ventilation_enabled
        return True

    def solver_dropdowns(self):
        a = self.app
        active = a.installation_header_state.active
        if active in ("cli", "coc"):
            definition = a.CLI_INSTALLATION if active == "cli" else a.COC_INSTALLATION
            labels = (
                definition.strategy_labels,
                definition.search_backend_labels,
                definition.heuristic_mode_labels,
                definition.graph_labels,
                definition.start_mode_labels,
            )
            state = getattr(a.workbench_state, active)
        else:
            labels = (
                a.demo_config.routing_strategies,
                a.demo_config.router_backends,
                a.demo_config.heuristic_modes,
                a.demo_config.graph_types,
                a.demo_config.room_start_modes,
            )
            state = a
        opened = a.workbench_state.solver_dropdown_open_for
        return tuple(
            SolverDropdown(
                key, label,
                tuple(DropdownOption(str(i), value) for i, value in enumerate(values)),
                str(getattr(state, attribute)),
                opened == (active, key),
            )
            for (key, label, attribute), values in zip(SOLVER_CONTROL_FIELDS, labels)
        )

    def handle_solver_dropdown_click(self, pos):
        a = self.app
        active = a.installation_header_state.active
        dropdowns = self.solver_dropdowns()
        opened = next((item for item in dropdowns if item.open), None)
        if opened is not None:
            index = next(i for i, item in enumerate(dropdowns) if item.key == opened.key)
            _control, option_bounds = solver_dropdown_bounds(index, len(opened.options))
            selected = next((int(option.key) for option, bounds in zip(opened.options, option_bounds)
                             if self.pygame.Rect(bounds).collidepoint(pos)), None)
            if selected is not None:
                attribute = next(item[2] for item in SOLVER_CONTROL_FIELDS if item[0] == opened.key)
                target = getattr(a.workbench_state, active) if active in ("cli", "coc") else a
                changed = getattr(target, attribute) != selected
                setattr(target, attribute, selected)
                if active == "cli":
                    if opened.key == "starts" and selected == 1 and target.router_backend_idx not in a.CLI_INSTALLATION.directed_backend_indices:
                        target.router_backend_idx = 4
                    elif opened.key == "router" and selected not in a.CLI_INSTALLATION.directed_backend_indices:
                        target.room_start_mode_idx = 0
                    elif opened.key == "router" and selected in a.CLI_INSTALLATION.directed_backend_indices:
                        target.room_start_mode_idx = 1
                a.workbench_state.solver_dropdown_open_for = None
                return True, changed, opened.key
        for index, dropdown in enumerate(dropdowns):
            control, _options = solver_dropdown_bounds(index, len(dropdown.options))
            if self.pygame.Rect(control).collidepoint(pos):
                if len(dropdown.options) <= 1:
                    return True, False, dropdown.key
                if hasattr(a, "installation_ui_session"):
                    a.installation_ui_session = a.installation_ui_session.dismiss_dropdown()
                key = (active, dropdown.key)
                a.workbench_state.solver_dropdown_open_for = (
                    None if a.workbench_state.solver_dropdown_open_for == key else key
                )
                return True, False, dropdown.key
        if opened is not None:
            a.workbench_state.solver_dropdown_open_for = None
        return False, False, None

    def handle_machine_selector_click(self, pos):
        a = self.app
        active = a.installation_header_state.active
        if active not in a.installation_ui_session.installations:
            return False, False
        options = a.installation_ui_session.options(active)
        if len(options) <= 1:
            return False, False
        state = a.installation_ui_session.dropdown_state(active)
        control, option_bounds = dropdown_bounds(
            25, MACHINE_SELECTOR_Y, a.canvas_ui.viewport.canvas_left - 70,
            len(options), height=24,
        )
        if self.pygame.Rect(control).collidepoint(pos):
            a.workbench_state.solver_dropdown_open_for = None
            a.installation_ui_session = a.installation_ui_session.toggle_dropdown(active)
            return True, False
        if state.open:
            selected = next((option.key for option, bounds in zip(options, option_bounds)
                             if self.pygame.Rect(bounds).collidepoint(pos)), None)
            if selected is not None:
                transition = a.installation_ui_session.choose(active, selected)
                a.installation_ui_session = transition.session
                dwelling = a.workbench_state.dwelling.active
                if active == "cli" and transition.changed and dwelling is not None:
                    placement = a.cli_runtime.place_machine(
                        dwelling.prepared, a.CLI_INSTALLATION.machine_catalog.value(selected),
                    )
                    a.workbench_state.installation("cli").pose.set(
                        placement.center, placement.angle_degrees,
                    )
                return True, transition.runtime_invalidation_required
            a.installation_ui_session = a.installation_ui_session.dismiss_dropdown()
        return False, False

    def draw_min_piece_slider(self, screen, font, x, y, width):
        a = self.app
        a.canvas_ui.controls.min_piece_reset = self.pygame.Rect(0, 0, 0, 0)
        if a.installation_header_state.active != "sal":
            a.canvas_ui.controls.min_piece = self.pygame.Rect(0, 0, 0, 0)
            text = "Piece validation: scope pending"
            if a.installation_header_state.active == "cli" and a.workbench_state.cli.router_backend_idx in a.CLI_INSTALLATION.length_backend_indices:
                state, definition = a.workbench_state.cli, a.CLI_INSTALLATION
                minimum = state.minimum_segment_length_mm
                automatic = minimum is None
                if automatic:
                    minimum = a.cli_runtime.last_result.supply_minimum_length_mm
                if minimum is not None:
                    a.canvas_ui.draw_weight_slider(
                        screen, font, x, y, width,
                        "Hard min (auto)" if automatic else "Hard min", minimum,
                        definition.minimum_segment_length_min_mm,
                        definition.minimum_segment_length_max_mm,
                        a.theme.text, a.theme.muted,
                        name="min_piece", suffix=" mm", integer=True,
                    )
                    return
                text = "Hard minimum: sizing unavailable"
            label = font.render(text, True, a.theme.muted)
            screen.blit(label, (x, y))
            return
        a.canvas_ui.draw_min_piece_slider(
            screen, font, x, y, width, a.min_piece_factor,
            a.demo_config.min_piece_factor_min, a.demo_config.min_piece_factor_max,
            a.theme.text, a.theme.muted,
        )

    def draw_weight_slider(self, screen, font, x, y, width, label, value, minimum, maximum,
                           color, rect_name, suffix="", integer=False):
        a = self.app
        a.canvas_ui.draw_weight_slider(
            screen, font, x, y, width, label, value, minimum, maximum,
            a.theme.text, a.theme.muted, name=rect_name, suffix=suffix, integer=integer,
        )

    def record_current_solution(self, routes, elapsed_ms, marker_label=None, marker_color=(241, 196, 15)):
        a = self.app
        if routes:
            a.observation.record_history(
                routes, a.count_segment_crossings(routes), elapsed_ms,
            )
            if marker_label:
                a.observation.add_history_marker(marker_label, marker_color)

    def handle_terminal_tool_button_click(self, pos):
        a = self.app
        if not active_capabilities(a).terminal_tools:
            a.preferred_terminal_tool_mode = None
            return None
        for mode, rect, _ in a.canvas_ui.terminal_buttons():
            if not rect.collidepoint(pos):
                continue
            if mode == "reset":
                a.preferred_terminal_tool_mode = None
                return "reset"
            if mode == "map":
                a.terminal_validity_overlay_enabled = not a.terminal_validity_overlay_enabled
                return "map"
            a.preferred_terminal_tool_mode = None if a.preferred_terminal_tool_mode == mode else mode
            return "mode"
        return None

    def draw_terminal_tool_buttons(self, screen, font_bold, font_small):
        a = self.app
        a.canvas_ui.draw_terminal_tools(
            screen, font_bold, font_small, active_mode=a.preferred_terminal_tool_mode,
            overlay_enabled=a.terminal_validity_overlay_enabled,
            colors=(a.theme.text, a.theme.muted, a.theme.terminal_allowed),
        )

    def dwelling_selector_options(self):
        return ("New random dwelling",) + tuple(case.label for case in self.app.REAL_DWELLING_SCENARIOS)

    def draw_canvas_tool_controls(self, screen, font, ruler_mode, dwelling_selector_open=False):
        a = self.app
        return a.canvas_ui.draw_toolbar(screen, font, state={
            "ruler_enabled": ruler_mode, "edge_weights_enabled": a.edge_weight_heatmap_enabled,
            "diameter_width_enabled": a.route_real_diameter_width_enabled,
            "duct_geometry_mode": a.workbench_state.cli.duct_geometry_mode_idx,
            "small_weight_view": a.edge_weight_view_mode_idx == 0,
            "dwelling_label": a.current_scenario_label,
            "dwelling_options": self.dwelling_selector_options(),
            "dwelling_selector_open": dwelling_selector_open,
            "active_terminal_mode": a.preferred_terminal_tool_mode, "text_color": a.theme.text,
        })

    def set_ruler_cursor(self, enabled):
        try:
            self.pygame.mouse.set_cursor(
                self.pygame.SYSTEM_CURSOR_CROSSHAIR if enabled else self.pygame.SYSTEM_CURSOR_ARROW,
            )
        except self.pygame.error:
            pass

    def draw_ruler_overlay(self, screen, font, start_mm, end_mm):
        a = self.app
        return a.canvas_ui.draw_ruler(screen, font, start_mm, end_mm, a.theme.text)

    def invalidate_room_start_node_cache(self):
        terminal_runtime = self.app.workbench_state.dwelling.terminal_runtime
        if terminal_runtime is not None:
            terminal_runtime.invalidate_candidates()

    def invalidate_terminal_validity_cache(self):
        self.app.terminal_ui.invalidate()

    def draw_colorbar(self, screen, node_scores):
        a = self.app
        return a.canvas_ui.draw_distance_colorbar(
            screen, bool(node_scores), a.heatmap_scale_mode, a.heatmap_palette_idx, a.theme.text,
        )

    def get_placeable_heatmap_scores(self, scores):
        return self.app.placement_session.placeable_scores(scores)

    def draw_distance_heatmap(self, screen, scores):
        a = self.app
        return a.canvas_ui.draw_distance_heatmap(
            screen, self.get_placeable_heatmap_scores(scores),
            base_env=a.workbench_state.dwelling.workspace.base_env,
            grid_spacing=a.demo_config.grid_spacing_mm,
            blocked_regions=a.workbench_state.dwelling.machine_vertical_clearance_blocks,
            scale_mode=a.heatmap_scale_mode, palette_index=a.heatmap_palette_idx,
            fill_color=a.theme.machine_clearance_fill,
            hatch_color=a.theme.machine_clearance_hatch,
        )

    def draw_edge_weight_heatmap(self, screen, workspace=None):
        a = self.app
        workspace = workspace or a.workbench_state.dwelling.workspace
        if a.edge_weight_heatmap_enabled:
            return a.canvas_ui.draw_edge_heatmap(
                screen, workspace.overlay.values, workspace.env,
                a.demo_config.overlap_block_weight, a.theme.blocked_edge,
            )

    def draw_edge_weight_colorbar(self, screen, workspace=None):
        a = self.app
        workspace = workspace or a.workbench_state.dwelling.workspace
        if a.edge_weight_heatmap_enabled:
            return a.canvas_ui.draw_edge_colorbar(
                screen, workspace.overlay.values,
                a.demo_config.overlap_block_weight,
                a.theme.blocked_edge, a.theme.text,
            )

    def get_route_draw_width(self, route_name):
        a = self.app
        if a.route_real_diameter_width_enabled:
            if route_name == "Kitchen Exhaust":
                result = a.workbench_state.installation("coc").result
                diameter = getattr(result, "diameter_mm", 125)
                return max(1, int(round(diameter * a.canvas_ui.viewport.scale)))
            return max(1, int(round(
                a.routing_controller.route_diameter(route_name)
                * a.canvas_ui.viewport.scale
            )))
        return 5 if a.SAL_INSTALLATION.is_large_route(route_name) else 3
