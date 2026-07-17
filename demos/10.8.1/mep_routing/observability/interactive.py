"""Interactive application adapter for routing observations and solution logs."""

from __future__ import annotations

from .runtime import RoutingObservationRuntime
from .snapshots import restored_snapshot_state
from ..ui.solution_logs import DEFAULT_BEST_METRICS, metric_value_for_log, solution_kpis


class InteractiveObservationController:
    """Own observation state while adapting the demo's live compatibility fields."""

    def __init__(self, app, chrome):
        self.app = app
        self.chrome = chrome
        self.runtime = RoutingObservationRuntime(
            route_length_m=app._total_route_length_m,
            route_turns=app.count_solution_turns,
            route_score=app.get_solution_score,
            solution_kpis=self.current_kpis,
            metric_value=metric_value_for_log,
            best_metrics=DEFAULT_BEST_METRICS,
        )

    @property
    def history(self):
        return self.runtime.history

    @property
    def logs(self):
        return self.runtime.logs

    def current_kpis(self, routes, elapsed_ms):
        a = self.app
        return solution_kpis(
            routes, elapsed_ms, a.count_segment_crossings, a._total_route_length_m,
            a.count_solution_turns, a.count_solution_short_pieces, a.get_solution_score,
        )

    def current_state(self):
        a = self.app
        terminals = a.workbench_state.dwelling.terminal_runtime
        sal_pose = a.workbench_state.installation("sal").pose
        return {
            "source_mode_idx": a.dwelling_source_idx,
            "real_scenario_idx": a.real_scenario_idx,
            "routing_frame_idx": a.routing_frame_idx,
            "scenario_label": a.current_scenario_label,
            "machine": (float(sal_pose.x_mm), float(sal_pose.y_mm), int(sal_pose.angle_degrees)),
            "graph_type_idx": a.graph_type_idx,
            "routing_strategy_idx": a.routing_strategy_idx,
            "router_backend_idx": a.router_backend_idx,
            "heuristic_mode_idx": a.heuristic_mode_idx,
            "rotation_mode_idx": a.rotation_mode_idx,
            "room_start_mode_idx": a.room_start_mode_idx,
            "weight_mode_idx": a.weight_mode_idx,
            "edge_weight_view_mode_idx": a.edge_weight_view_mode_idx,
            "route_real_diameter_width_enabled": bool(a.route_real_diameter_width_enabled),
            "min_piece_factor": float(a.min_piece_factor),
            "bend_weight": float(a.C_BEND),
            "crossing_penalty_multiplier": float(a.crossing_penalty_multiplier),
            "preferred_terminal_points_by_room": {} if terminals is None else terminals.preferred_points_by_room,
            "preferred_terminal_areas": [] if terminals is None else terminals.preferred_areas,
        }

    def record_history(self, routes, crossings_count, elapsed_ms):
        self.runtime.record(routes, crossings_count, elapsed_ms)
        if routes:
            self.runtime.update_auto_bests(self.current_state(), routes, "Auto best", elapsed_ms, 0)

    def clear_history_buffers(self):
        self.runtime.clear()

    def log_current_solution(self, routes, status, elapsed_ms, total_nodes):
        return self.runtime.log_manual(self.current_state(), routes, status, elapsed_ms, total_nodes)

    def restore_solution_log(self, entry):
        a = self.app
        state = restored_snapshot_state(entry, a.C_BEND_DEFAULT, a.CROSSING_MULTIPLIER_DEFAULT)
        for name in (
            "graph_type_idx", "routing_strategy_idx", "router_backend_idx", "heuristic_mode_idx",
            "rotation_mode_idx", "room_start_mode_idx", "weight_mode_idx", "edge_weight_view_mode_idx",
            "route_real_diameter_width_enabled", "min_piece_factor",
        ):
            setattr(a, name, state[name])
        sal_pose = a.workbench_state.installation("sal").pose
        sal_pose.set(state["machine"][:2], state["machine"][2])
        a.C_BEND = state["bend_weight"]
        a.crossing_penalty_multiplier = state["crossing_penalty_multiplier"]
        terminal_runtime = a.workbench_state.dwelling.terminal_runtime
        if terminal_runtime is not None:
            terminal_runtime.restore_preferences(
                state["preferred_terminal_points_by_room"], state["preferred_terminal_areas"],
            )
        self.logs.selected_log_id = state["id"]
        a.build_grid(machine_pins=a.get_machine_pins(*sal_pose.as_tuple()))
        return a.solve_ventilation_routing()

    def handle_solution_log_click(self, pos):
        return self.chrome.solution_log_action(pos)

    def draw_solution_logs_panel(self, screen, font_small, font_bold):
        self.chrome.draw_solution_logs(screen, font_small, font_bold, self.logs)

    def draw_plots(self, screen, font_small, font_title, font_value, font_minimum):
        return self.chrome.draw_plots(
            screen, font_small, font_title, font_value, font_minimum, self.history,
        )

    def draw_card_help_button(self, screen, card_id, rect, font):
        self.chrome.draw_help_button(screen, card_id, rect, font)

    def draw_help_popup(self, screen, font):
        return self.chrome.draw_help_popup(screen, font)

    def set_transient_message(self, text, duration_ms=2400):
        self.chrome.show_message(text, duration_ms)

    def draw_transient_message(self, screen, font):
        return self.chrome.draw_transient(screen, font)

    def draw_viewer_legend(self, screen, font):
        return self.chrome.draw_viewer_legend(
            screen, font, self.app.terminal_validity_overlay_enabled,
        )
