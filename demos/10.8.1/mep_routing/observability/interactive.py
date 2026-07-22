"""Interactive application adapter for routing observations and solution logs."""

from __future__ import annotations

from .runtime import RoutingObservationRuntime
from .session import RoutingHistory
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
            route_score=app.routing_controller.solution_score,
            solution_kpis=self.current_kpis,
            metric_value=metric_value_for_log,
            best_metrics=DEFAULT_BEST_METRICS,
        )
        self._histories = {"all": self.runtime.history}
        self.kpi_scope = "all"

    @property
    def history(self):
        return self.runtime.history

    @property
    def displayed_history(self):
        """History selected by the KPI rail; solution logs always remain aggregate."""
        return self._histories.get(self.kpi_scope, self.history)

    @property
    def logs(self):
        return self.runtime.logs

    def current_kpis(self, routes, elapsed_ms, *, installation_key=None, score_override=None):
        a = self.app
        score_fn = (
            (lambda current_routes, crossings: score_override)
            if score_override is not None
            else (
                lambda current_routes, crossings: a.routing_controller.installation_solution_score(
                    installation_key, current_routes, crossings,
                )
            ) if installation_key is not None else a.routing_controller.solution_score
        )
        return solution_kpis(
            routes, elapsed_ms, a.count_segment_crossings, a._total_route_length_m,
            a.count_solution_turns, a.routing_controller.count_short_pieces,
            score_fn,
        )

    def kpi_scope_options(self):
        """Expose aggregate plus installations that have yielded a plot sample."""
        labels = {
            item.key: item.label
            for item in getattr(self.app, "INSTALLATION_OPTIONS", ())
        }
        colors = getattr(getattr(self.app, "theme", None), "installation_colors", {})
        return tuple(
            (key, "All installations" if key == "all" else labels.get(key, key.title()),
             (82, 133, 203) if key == "all" else colors.get(key, (105, 113, 124)))
            for key in self._histories
        )

    def select_kpi_scope(self, key):
        if key not in self._histories:
            return False
        self.kpi_scope = key
        return True

    def handle_kpi_scope_click(self, pos):
        from ..ui.plots import kpi_scope_hit
        viewport = self.chrome.viewport
        key = kpi_scope_hit(
            pos, viewport.window_width, self.chrome.panel_width, self.kpi_scope_options(),
        )
        return key is not None and self.select_kpi_scope(key)

    def _history_for(self, key):
        return self._histories.setdefault(key, RoutingHistory(maxlen=self.history.maxlen))

    @staticmethod
    def _combined_routes(runs):
        return tuple(
            route
            for result in runs.values()
            for route in (getattr(result, "routes", ()) or ())
        )

    def _record_sample(self, key, routes, elapsed_ms):
        history = self._history_for(key)
        history.append(self.current_kpis(routes, elapsed_ms, installation_key=key))
        return history

    def _aggregate_score(self, runs):
        total = 0.0
        for key, result in runs.items():
            routes = getattr(result, "routes", ()) or ()
            crossings = self.app.count_segment_crossings(routes) if routes else 0
            total += self.app.routing_controller.installation_solution_score(
                key, routes, crossings,
            )
        return total

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
        runs = self.app.workbench_state.runs
        if runs:
            all_routes = self._combined_routes(runs)
            all_elapsed_ms = sum(float(getattr(result, "elapsed_ms", 0.0)) for result in runs.values())
            aggregate_score = self._aggregate_score(runs)
            for key, result in runs.items():
                self._record_sample(key, getattr(result, "routes", ()) or (), getattr(result, "elapsed_ms", 0.0))
        else:
            all_routes, all_elapsed_ms = routes, elapsed_ms
            aggregate_score = None
        history = self._history_for("all")
        history.append(self.current_kpis(
            all_routes, all_elapsed_ms, score_override=aggregate_score,
        ))
        if all_routes:
            self.runtime.update_auto_bests(self.current_state(), all_routes, "Auto best", all_elapsed_ms, 0)

    def add_history_marker(self, label, color):
        for history in self._histories.values():
            history.add_marker(label, color)

    def clear_history_buffers(self):
        self.runtime.clear()
        for key, history in self._histories.items():
            if key != "all":
                history.clear()
        self._histories = {"all": self.history}
        self.kpi_scope = "all"

    def log_current_solution(self, routes, status, elapsed_ms, total_nodes):
        if not routes:
            return False
        self.record_history(routes, self.app.count_segment_crossings(routes), elapsed_ms)
        return self.runtime.log_manual(
            self.current_state(), routes, status, elapsed_ms, total_nodes, record=False,
        )

    def restore_solution_log(self, entry):
        a = self.app
        state = restored_snapshot_state(
            entry, a.demo_config.bend_default, a.demo_config.crossing_default,
        )
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
        a.routing_controller.build_grid(
            machine_pins=a.routing_controller.machine_pins(*sal_pose.as_tuple()),
        )
        return a.routing_controller.solve_installations()

    def handle_solution_log_click(self, pos):
        return self.chrome.solution_log_action(pos)

    def draw_solution_logs_panel(self, screen, font_small, font_bold):
        self.chrome.draw_solution_logs(screen, font_small, font_bold, self.logs)

    def draw_plots(self, screen, font_small, font_title, font_value, font_minimum):
        return self.chrome.draw_plots(
            screen, font_small, font_title, font_value, font_minimum, self.displayed_history,
            self.kpi_scope, self.kpi_scope_options(),
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
        options = {item.key: item.label for item in self.app.INSTALLATION_OPTIONS}
        entries = tuple(
            (
                options.get(key, key.title()),
                self.app.theme.installation_colors.get(key, (105, 113, 124)),
                key == self.app.installation_header_state.active,
            )
            for key in self.app.installation_header_state.order
            if key in self.app.installation_header_state.enabled
        )
        return self.chrome.draw_viewer_legend(
            screen, font, self.app.terminal_validity_overlay_enabled, entries,
        )
