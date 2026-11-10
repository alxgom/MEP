"""Application of workbench UI commands to live interactive state."""

from __future__ import annotations

from .events import RoutingKeyTransition, routing_key_transition


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
                elif action == "duct_geometry":
                    if a.installation_header_state.active == "cli":
                        state = a.workbench_state.cli
                        state.duct_geometry_mode_idx = (state.duct_geometry_mode_idx + 1) % 3
                    else:
                        a.observation.set_transient_message("Duct geometry modes are available for Cli")
                return action
        if a.canvas_ui.weight_switch_rect().collidepoint(pos):
            return "weight_view"
        return None

    def set_min_piece_from_slider_x(self, x):
        a = self.app
        if a.installation_header_state.active == "cli":
            state, definition = a.workbench_state.cli, a.CLI_INSTALLATION
            if state.router_backend_idx not in definition.length_backend_indices:
                return
            raw = a.canvas_ui.slider_value(
                "min_piece", x, definition.minimum_segment_length_factor_min,
                definition.minimum_segment_length_factor_max,
                state.minimum_segment_length_factor,
            )
            step = definition.minimum_segment_length_factor_step
            state.minimum_segment_length_factor = round(max(
                definition.minimum_segment_length_factor_min,
                min(definition.minimum_segment_length_factor_max, round(raw / step) * step),
            ), 2)
            return
        if a.installation_header_state.active != "sal":
            return
        a.min_piece_factor = a.canvas_ui.slider_value(
            "min_piece", x, a.demo_config.min_piece_factor_min,
            a.demo_config.min_piece_factor_max, a.min_piece_factor,
        )

    def reset_minimum_segment_length(self):
        if self.app.installation_header_state.active == "cli":
            self.app.workbench_state.cli.minimum_segment_length_factor = 1.0

    def set_yellow_placement_penalty_from_slider_x(self, x):
        a = self.app
        state = a.workbench_state.cli
        state.yellow_placement_penalty_mm = round(a.canvas_ui.slider_value(
            "yellow_placement_penalty", x, 0, a.CLI_INSTALLATION.yellow_placement_penalty_max_mm,
            state.yellow_placement_penalty_mm,
        ))

    def set_bend_weight_from_slider_x(self, x):
        a = self.app
        if a.installation_header_state.active == "cli":
            state = a.workbench_state.cli
            raw = a.canvas_ui.slider_value(
                "bend", x, a.demo_config.bend_min, a.demo_config.bend_max,
                state.bend_penalty_mm,
            )
            state.bend_penalty_mm = float(round(raw / 100.0) * 100)
            return
        raw = a.canvas_ui.slider_value(
            "bend", x, a.demo_config.bend_min, a.demo_config.bend_max, a.C_BEND,
        )
        a.C_BEND = float(round(raw / 100.0) * 100)

    def set_crossing_weight_from_slider_x(self, x):
        a = self.app
        if a.installation_header_state.active == "cli":
            state = a.workbench_state.cli
            state.crossing_penalty_multiplier = a.canvas_ui.slider_value(
                "crossing", x, a.demo_config.crossing_min,
                a.demo_config.crossing_max, state.crossing_penalty_multiplier,
            )
            return
        a.crossing_penalty_multiplier = a.canvas_ui.slider_value(
            "crossing", x, a.demo_config.crossing_min, a.demo_config.crossing_max,
            a.crossing_penalty_multiplier,
        )

    def reset_bend_weight(self):
        if self.app.installation_header_state.active == "cli":
            self.app.workbench_state.cli.bend_penalty_mm = 4000.0
        else:
            self.app.C_BEND = self.app.demo_config.bend_default

    def reset_crossing_weight(self):
        if self.app.installation_header_state.active == "cli":
            self.app.workbench_state.cli.crossing_penalty_multiplier = 5.0
        else:
            self.app.crossing_penalty_multiplier = self.app.demo_config.crossing_default

    def apply_routing_key_command(self, event_key):
        a, p = self.app, self.pygame
        active = a.installation_header_state.active
        if active not in a.installation_header_state.enabled:
            return None
        if active == "cli":
            return self._apply_cli_routing_key_command(event_key)
        if active == "coc":
            return self._apply_coc_routing_key_command(event_key)
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

    def _apply_cli_routing_key_command(self, event_key):
        a, p = self.app, self.pygame
        state = a.workbench_state.cli
        definition = a.CLI_INSTALLATION
        if event_key == p.K_p:
            state.auto_placement_mode_idx = (
                state.auto_placement_mode_idx + 1
            ) % len(definition.auto_placement_labels)
            return RoutingKeyTransition({}, solve=False, record_history=False)
        if event_key == p.K_a:
            if state.auto_placement_mode_idx == 0:
                state.auto_placement_mode_idx = definition.virtual_placement_mode_index
            return RoutingKeyTransition({}, solve=False, record_history=False, needs_auto_placement=True)
        controls = {
            p.K_c: ("routing_strategy_idx", len(definition.strategy_labels), "Strategy"),
            p.K_l: ("router_backend_idx", len(definition.search_backend_labels), "Router"),
            p.K_y: ("heuristic_mode_idx", len(definition.heuristic_mode_labels), "Heuristic"),
            p.K_TAB: ("graph_type_idx", len(definition.graph_labels), "Grid"),
            p.K_t: ("room_start_mode_idx", len(definition.start_mode_labels), "Starts"),
        }
        selected = controls.get(event_key)
        if selected is None:
            return None
        attribute, count, label = selected
        value = (getattr(state, attribute) + 1) % count
        setattr(state, attribute, value)
        if attribute == "router_backend_idx":
            state.room_start_mode_idx = 1 if value in definition.directed_backend_indices else 0
        elif attribute == "room_start_mode_idx" and value == 1 and state.router_backend_idx not in definition.directed_backend_indices:
            state.router_backend_idx = 4
        return RoutingKeyTransition(
            {}, markers=((f"{label}:{value}", (52, 152, 219)),),
        )

    def _apply_coc_routing_key_command(self, event_key):
        a, p = self.app, self.pygame
        state, definition = a.workbench_state.coc, a.COC_INSTALLATION
        controls = {
            p.K_c: ("routing_strategy_idx", len(definition.strategy_labels), "Strategy"),
            p.K_l: ("router_backend_idx", len(definition.search_backend_labels), "Router"),
            p.K_y: ("heuristic_mode_idx", len(definition.heuristic_mode_labels), "Heuristic"),
            p.K_TAB: ("graph_type_idx", len(definition.graph_labels), "Grid"),
            p.K_t: ("room_start_mode_idx", len(definition.start_mode_labels), "Starts"),
        }
        selected = controls.get(event_key)
        if selected is None:
            return None
        attribute, count, label = selected
        value = (getattr(state, attribute) + 1) % count
        setattr(state, attribute, value)
        return RoutingKeyTransition(
            {}, markers=((f"{label}:{value}", (230, 144, 52)),),
        )
