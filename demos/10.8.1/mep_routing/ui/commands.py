"""Application of workbench UI commands to live interactive state."""

from __future__ import annotations

from .events import routing_key_transition


class WorkbenchCommandController:
    """Keep Pygame command decoding and its state effects out of the entrypoint."""

    def __init__(self, app, pygame):
        self.app = app
        self.pygame = pygame

    def handle_canvas_tool_button_click(self, pos):
        a = self.app
        for action, rect, _ in a.canvas_ui.toolbar_buttons():
            if rect.collidepoint(pos):
                a.canvas_ui.apply_view_action(action)
                if action == "diameter":
                    a.route_real_diameter_width_enabled = not a.route_real_diameter_width_enabled
                return action
        if a.canvas_ui.weight_switch_rect().collidepoint(pos):
            return "weight_view"
        return None

    def set_min_piece_factor_from_slider_x(self, x):
        a = self.app
        a.min_piece_factor = a.canvas_ui.slider_value(
            "min_piece", x, a.demo_config.min_piece_factor_min,
            a.demo_config.min_piece_factor_max, a.min_piece_factor,
        )

    def set_bend_weight_from_slider_x(self, x):
        a = self.app
        raw = a.canvas_ui.slider_value(
            "bend", x, a.demo_config.bend_min, a.demo_config.bend_max, a.C_BEND,
        )
        a.C_BEND = float(round(raw / 100.0) * 100)

    def set_crossing_weight_from_slider_x(self, x):
        a = self.app
        a.crossing_penalty_multiplier = a.canvas_ui.slider_value(
            "crossing", x, a.demo_config.crossing_min, a.demo_config.crossing_max,
            a.crossing_penalty_multiplier,
        )

    def reset_bend_weight(self):
        self.app.C_BEND = self.app.demo_config.bend_default

    def reset_crossing_weight(self):
        self.app.crossing_penalty_multiplier = self.app.demo_config.crossing_default

    def apply_routing_key_command(self, event_key):
        a, p = self.app, self.pygame
        sal_pose = a.workbench_state.installation("sal").pose
        command = {
            p.K_r: "rotate_machine", p.K_c: "cycle_strategy",
            p.K_t: "cycle_room_start", p.K_l: "cycle_router_backend",
            p.K_y: "cycle_heuristic", p.K_TAB: "cycle_graph",
            p.K_a: "toggle_auto_placement", p.K_p: "cycle_auto_placement",
            p.K_u: "cycle_rotation_mode", p.K_w: "cycle_weight_mode",
        }.get(event_key)
        if command is None:
            return None
        transition = routing_key_transition(command, {
            "machine_angle": sal_pose.angle_degrees,
            "auto_placement_mode_idx": a.auto_placement_mode_idx,
            "routing_strategy_idx": a.routing_strategy_idx,
            "room_start_mode_idx": a.room_start_mode_idx,
            "router_backend_idx": a.router_backend_idx,
            "heuristic_mode_idx": a.heuristic_mode_idx,
            "graph_type_idx": a.graph_type_idx,
            "rotation_mode_idx": a.rotation_mode_idx,
            "weight_mode_idx": a.weight_mode_idx,
        }, {
            "routing_strategy": len(a.demo_config.routing_strategies),
            "room_start": len(a.demo_config.room_start_modes),
            "router_backend": len(a.demo_config.router_backends),
            "heuristic": len(a.demo_config.heuristic_modes),
            "graph": len(a.demo_config.graph_types),
            "rotation_mode": len(a.demo_config.rotation_modes),
            "auto_placement": len(a.demo_config.auto_placement_modes), "weight_mode": 2,
        })
        if transition is None:
            return None
        for name, value in transition.state.items():
            if name == "machine_angle":
                sal_pose.angle_degrees = value
            else:
                setattr(a, name, value)
        if transition.rebuild_graph:
            a.routing_controller.build_grid(
                machine_pins=a.routing_controller.machine_pins(*sal_pose.as_tuple()),
            )
        if transition.apply_rotation_mode:
            a.routing_controller.apply_rotation_mode()
        workspace = a.workbench_state.dwelling.workspace
        if transition.refresh_placement_fields and workspace.base_env is not None and a.shaft_extraction is not None:
            shaft_nodes, _ = a.routing_controller.shaft_entry_nodes(
                workspace.base_env, workspace.base_spatial_index,
            )
            a.placement_session.weight_mode = a.weight_mode_idx
            a.ap_scores, a.ap_fields = a.placement_session.placement_scores(
                workspace.base_env, workspace.base_spatial_index, shaft_nodes,
            )
        return transition
