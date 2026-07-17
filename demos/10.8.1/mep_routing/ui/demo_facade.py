"""Thin UI facade for Demo 10.8.1's remaining callback surface."""

from __future__ import annotations

from .dropdown import dropdown_bounds
from .installation_header import installation_header_layout
from .installation_projection import active_capabilities
from .sidebar import MACHINE_SELECTOR_Y


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
                a.observation.history.add_marker(marker_label, marker_color)

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

    def draw_edge_weight_heatmap(self, screen):
        a = self.app
        if a.edge_weight_heatmap_enabled:
            return a.canvas_ui.draw_edge_heatmap(
                screen, a.workbench_state.dwelling.workspace.overlay.values,
                a.workbench_state.dwelling.workspace.env,
                a.demo_config.overlap_block_weight, a.theme.blocked_edge,
            )

    def draw_edge_weight_colorbar(self, screen):
        a = self.app
        if a.edge_weight_heatmap_enabled:
            return a.canvas_ui.draw_edge_colorbar(
                screen, a.workbench_state.dwelling.workspace.overlay.values,
                a.demo_config.overlap_block_weight,
                a.theme.blocked_edge, a.theme.text,
            )

    def get_route_draw_width(self, route_name):
        a = self.app
        if a.route_real_diameter_width_enabled:
            return max(1, int(round(
                a.routing_controller.route_diameter(route_name)
                * a.canvas_ui.viewport.scale
            )))
        return 5 if a.SAL_INSTALLATION.is_large_route(route_name) else 3
