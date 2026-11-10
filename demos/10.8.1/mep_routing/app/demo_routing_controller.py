"""Demo application routing facade kept separate from the Pygame entrypoint."""

from __future__ import annotations

from types import SimpleNamespace

from shapely.ops import unary_union

from mep_routing.app.installation_context import installation_context_obstacles
from mep_routing.app.installation_execution import InstallationCoordinator
from mep_routing.app.installation_runtime import (
    CliInstallationRuntime, CocInstallationRuntime, InstallationRegistry,
    SalInstallationRuntime,
)
from mep_routing.domain import machine_pins, port_access_specs
from mep_routing.installations.sal import (
    SalInteractiveCallbacks,
    SalInteractiveSolver,
    SalLiveRoutingSession,
    SalRouteMaterializer,
    SalSolverSettings,
)
from mep_routing.routing import (
    block_terminal_node_edges,
    count_segment_crossings,
    count_solution_turns,
    find_route_at_point,
    find_route_hit_at_point,
    line_graph_dir_from_points,
    selected_pin_names,
    terminal_node_indices,
    weighted_edge_cost,
)


class DemoRoutingController:
    """Own solver composition while the entrypoint supplies live demo state.

    ``app`` is the explicit compatibility boundary during the ongoing state-object
    migration.  All mutations performed here are application outcomes (pose,
    placement caches, and installation results), rather than hidden module state.
    """

    def __init__(self, app):
        self.app = app
        self.installations = InstallationRegistry((
            SalInstallationRuntime(self), CliInstallationRuntime(app),
            CocInstallationRuntime(app),
        ))
        self.installation_coordinator = InstallationCoordinator(
            self.installations, app.workbench_state,
        )

    def _pose(self, key="sal"):
        return self.app.workbench_state.installation(key).pose

    def _dwelling(self):
        return self.app.workbench_state.dwelling

    def machine_pins(self, cx, cy, angle_degrees):
        a = self.app
        return machine_pins(a.SAL_INSTALLATION.default_machine, cx, cy, angle_degrees)

    def port_access_specs(self, global_pins, machine_angle):
        a = self.app
        return port_access_specs(a.SAL_INSTALLATION.default_machine, global_pins, machine_angle)

    def shaft_entry_geometry(self, node_idx, env=None):
        a = self.app
        state = self._dwelling()
        env = env or state.workspace.env
        return state.active.shaft_entries.geometry(node_idx, env) if state.active else None

    def shaft_entry_nodes(self, env, kd=None):
        a = self.app
        dwelling = self._dwelling().active
        if dwelling is None:
            return [], None
        return dwelling.shaft_entries.select(
            env, spatial_index=kd, search_radius_mm=a.demo_config.shaft_entry_search_mm,
            grid_spacing_mm=a.demo_config.grid_spacing_mm,
            max_candidates=a.demo_config.shaft_entry_max_candidates,
        )

    def add_shaft_entry_segments(self, segments, first_node_idx):
        a = self.app
        state = self._dwelling()
        if state.active:
            state.active.shaft_entries.append_segments(
                segments, first_node_idx, state.workspace.env,
            )

    def update_dynamic_env(self, machine_polygon, context_obstacles=()):
        a = self.app
        pose = self._pose()
        pins = self.machine_pins(*pose.as_tuple())
        obstacle = unary_union((machine_polygon, *context_obstacles)) if context_obstacles else machine_polygon
        self._dwelling().graph_build_session.apply_machine_obstacle(
            obstacle, pins, pose.angle_degrees, clearance_mm=a.demo_config.machine_clearance_mm,
        )

    def build_grid(self, machine_pins=None):
        a = self.app
        pose = self._pose()
        pins = machine_pins or self.machine_pins(*pose.as_tuple())
        return self._dwelling().graph_build_session.build(
            a.graph_type_idx, pins, pose.angle_degrees, shift_hannan_walls=True,
        )

    def solver_settings(self):
        a = self.app
        return SalSolverSettings(
            bend_cost=a.C_BEND,
            crossing_penalty_multiplier=a.crossing_penalty_multiplier,
            duct_buffer_ratio=a.demo_config.duct_buffer_ratio,
            shaft_clearance_mm=a.demo_config.shaft_clearance_mm,
            machine_clearance_soft_margin_mm=a.demo_config.machine_clearance_soft_margin_mm,
            overlap_block_weight=a.demo_config.overlap_block_weight,
            min_piece_factor=a.min_piece_factor,
            heuristic_mode=a.heuristic_mode_idx,
            routing_strategy=a.routing_strategy_idx,
        )

    def live_session(self):
        a = self.app
        dwelling = self._dwelling().active
        pose = self._pose()
        return SalLiveRoutingSession(
            installation=a.SAL_INSTALLATION,
            machine_spec=a.SAL_INSTALLATION.default_machine,
            workspace=dwelling.workspace,
            routing_region=dwelling.routing_region,
            rooms=tuple(dwelling.rooms),
            walls=tuple(dwelling.walls),
            wall_polygons=tuple(dwelling.wall_polygons),
            shafts=tuple(dwelling.shafts),
            machine_center=pose.center,
            machine_angle=pose.angle_degrees,
            settings=self.solver_settings(),
            search_backend_index=a.router_backend_idx,
            estimate_turns=a.estimate_turns,
        )

    def route_analysis(self, policy=None):
        a = self.app
        return self.live_session().route_analysis(a.shaft_extraction, policy)

    def routing_runtime(self, env, policy=None):
        return self.live_session().routing_runtime(env, policy)

    def route_diameter(self, route_name):
        return self.route_analysis().route_diameter(route_name)

    def refresh_edge_weight_overlay(self, routes, workspace=None):
        a = self.app
        workspace = workspace or self._dwelling().workspace
        if workspace.env is None:
            workspace.overlay.reset()
            return
        if a.installation_header_state.active == "cli":
            return
        machine = a.SAL_INSTALLATION.default_machine
        diameter = machine.small_duct_diameter_mm if a.edge_weight_view_mode_idx == 0 else machine.large_duct_diameter_mm
        self.routing_runtime(workspace.env).refresh_edge_weight_overlay(routes, diameter)

    def count_short_pieces(self, routes):
        return self.route_analysis().count_short_pieces(routes)

    def route_at_point(self, routes, world_point):
        a = self.app
        return find_route_at_point(routes, world_point, max(40.0, 8.0 / a.canvas_ui.viewport.scale))

    def route_hit_at_point(self, routes, world_point):
        a = self.app
        return find_route_hit_at_point(routes, world_point, max(40.0, 8.0 / a.canvas_ui.viewport.scale))

    def selected_pin_names(self, route_name, routes, global_pins):
        return selected_pin_names(route_name, routes, global_pins)

    def route_materializer(self, route_plan):
        a = self.app
        workspace = self._dwelling().active.workspace
        return SalRouteMaterializer(
            workspace.env.nodes, workspace.spatial_index, route_plan,
            self.add_shaft_entry_segments if a.shaft_extraction else None,
        )

    def solution_score(self, routes, crossings, *, policy=None):
        return self.route_analysis(policy).score(routes, crossings)

    def installation_solution_score(self, installation_key, routes, crossings=None):
        if installation_key == "cli":
            from mep_routing.installations.cli import CliRouteAnalysis

            state = self.app.workbench_state.cli
            from mep_routing.graphs import GraphMode
            return CliRouteAnalysis(
                state.bend_penalty_mm,
                state.crossing_penalty_multiplier,
                include_diagonal=self.app.CLI_INSTALLATION.graph_modes[state.graph_type_idx] in (
                    GraphMode.MEDIAL_AXIS, GraphMode.HANNAN_MEDIAL),
            ).score(routes, crossings)
        return self.solution_score(routes, crossings)

    def validation_warnings(self, routes):
        a = self.app
        warnings = self.route_analysis().quality_warnings(routes)
        if (a.shaft_extraction is not None
                and a.demo_config.dwelling_source_modes[a.dwelling_source_idx] == "Real DB"
                and not a.shaft_core_entry_specs):
            warnings.append("missing core shaft entry metadata")
        return warnings

    def ensure_placement_scores(self):
        a = self.app
        a.placement_session.ensure_scores()
        a.ap_scores, a.ap_fields = a.placement_session.scores, a.placement_session.fields

    def align_rotation(self):
        a = self.app
        pose = self._pose()
        angle, selected, scores = a.placement_session.align_rotation(
            pose.center, pose.angle_degrees,
        )
        pose.angle_degrees = angle
        a.rotation_field_scores = a.placement_session.rotation_scores
        return selected, scores

    def apply_rotation_mode(self):
        a = self.app
        pose = self._pose()
        pose.angle_degrees = a.placement_session.apply_rotation_mode(
            a.rotation_mode_idx, pose.center, pose.angle_degrees,
        )
        a.rotation_field_scores = a.placement_session.rotation_scores

    def auto_place(self):
        a = self.app
        if a.installation_header_state.active == "cli":
            state = a.workbench_state.cli
            if state.auto_placement_mode_idx == 0:
                return
            dwelling = a.workbench_state.dwelling.active
            if dwelling is None:
                return
            machine_key = a.installation_ui_session.selected_keys["cli"]
            machine = a.CLI_INSTALLATION.machine_catalog.value(machine_key)
            pose = a.workbench_state.installation("cli").pose
            if state.auto_placement_mode_idx == a.CLI_INSTALLATION.virtual_placement_mode_index:
                header = a.installation_header_state
                runs = a.workbench_state.runs
                context = installation_context_obstacles(
                    SimpleNamespace(routes=runs[key].routes, machine_pins=self.installations.get(key).machine_pins(
                        a.workbench_state.installation(key),
                    ))
                    for key in header.order[:header.order.index("cli")] if key in header.enabled and key in runs
                )
                mask = self.installations.get("cli").placement_mask(dwelling.prepared, pose.angle_degrees)
                outcome = a.cli_runtime.place_machine_steiner(
                    dwelling.prepared, machine, pose.angle_degrees, mask, context_obstacles=context,
                    minimum_segment_length_factor=state.minimum_segment_length_factor,
                    bend_penalty_mm=state.bend_penalty_mm,
                    yellow_placement_penalty_mm=state.yellow_placement_penalty_mm,
                )
                message = (f"{outcome.status} at {pose.angle_degrees:g}° · {outcome.ranked}/{outcome.feasible} ranked "
                           f"· {outcome.verified} checked · {outcome.elapsed_ms:.0f} ms")
                a.observation.set_transient_message(message)
                if outcome.placement is not None:
                    pose.set(outcome.placement.center, outcome.placement.angle_degrees)
                    state.graph_type_idx = a.CLI_INSTALLATION.graph_modes.index(a.cli_runtime.virtual_placement.graph_mode)
                    state.router_backend_idx = a.cli_runtime.virtual_placement.backend_index
                    state.room_start_mode_idx = 1
                return outcome
            placement = a.cli_runtime.place_machine(
                dwelling.prepared, machine, minimum_segment_length_factor=state.minimum_segment_length_factor,
                angle_degrees=pose.angle_degrees,
            )
            if placement.feasible_candidates:
                pose.set(placement.center)
            else:
                a.observation.set_transient_message(f"No valid CLI machine position at {pose.angle_degrees:g}°")
            return
        a.placement_session.weight_mode = a.weight_mode_idx
        outcome, elapsed_ms = a.placement_session.auto_place(a.auto_placement_mode_idx)
        if outcome is None:
            return
        a.ap_scores, a.ap_fields = a.placement_session.scores, a.placement_session.fields
        if outcome.position is None:
            print("[Auto-Placement] No feasible machine placement satisfies the active restrictions")
            return
        pose = self._pose()
        pose.set(outcome.position, outcome.rotation)
        self.build_grid(machine_pins=self.machine_pins(*pose.as_tuple()))
        if a.auto_placement_mode_idx == 2:
            print(f"[Core-like Machine Placement] tried {outcome.candidate_count} feasible candidates in {elapsed_ms:.1f}ms")
        else:
            print(f"[Auto-Placement] Solved position {pose.center} at rotation {pose.angle_degrees} in {elapsed_ms:.2f}ms")

    def interactive_solver(self, context_obstacles=()):
        a = self.app
        dwelling = self._dwelling().active
        workspace = dwelling.workspace
        callbacks = SalInteractiveCallbacks(
            update_dynamic_env=lambda polygon: self.update_dynamic_env(polygon, context_obstacles),
            build_grid=self.build_grid,
            shaft_entry_nodes=lambda: self.shaft_entry_nodes(workspace.env, workspace.spatial_index),
            terminal_nodes=lambda _pins, shaft_idx, shaft_route: terminal_node_indices(
                dwelling.terminal_points, shaft_idx, workspace.spatial_index,
                shaft_route_name=shaft_route,
            ),
            block_terminal_edges=lambda weights, terminal_map, block_weight: block_terminal_node_edges(
                weights, workspace.env.adj, terminal_map, block_weight,
            ),
            weighted_edge_cost=weighted_edge_cost,
            line_graph_direction=line_graph_dir_from_points,
            route_start_nodes=lambda name: a.terminal_ui.route_start_nodes(
                name, use_nearest_terminal=(a.room_start_mode_idx == 1),
            ),
            count_crossings=count_segment_crossings,
        )
        return SalInteractiveSolver(
            live_session=self.live_session(),
            terminals=dwelling.terminal_points,
            graph_type=a.graph_type_idx,
            columns=(*dwelling.columns, *context_obstacles),
            blocked_vertical_regions=a.placement_session.clearance_blocks,
            route_materializer_factory=self.route_materializer,
            shaft_extraction=a.shaft_extraction,
            callbacks=callbacks,
        )

    def solve_installations(self):
        a = self.app
        execution = self.installation_coordinator.execute(
            a.installation_header_state.order, a.installation_header_state.enabled,
            a.installation_header_state.active, self._dwelling().active,
        )
        return execution.routing_tuple()
