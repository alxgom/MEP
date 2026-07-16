"""Interactive Sal solve facade over the live routing application boundary."""

from __future__ import annotations

from dataclasses import dataclass

from mep_routing.routing import MachineRoutingSession

from .live_solver import SalLiveRoutingCallbacks


@dataclass(frozen=True)
class SalInteractiveCallbacks:
    update_dynamic_env: object
    build_grid: object
    shaft_entry_nodes: object
    terminal_nodes: object
    block_terminal_edges: object
    weighted_edge_cost: object
    line_graph_direction: object
    route_start_nodes: object
    count_crossings: object


@dataclass(frozen=True)
class SalInteractiveSolver:
    """Build and execute one Sal solve from an interactive-state snapshot."""

    live_session: object
    terminals: object
    graph_type: int
    columns: tuple
    blocked_vertical_regions: tuple
    route_materializer_factory: object
    shaft_extraction: object
    callbacks: SalInteractiveCallbacks

    def solve(self):
        machine_session = MachineRoutingSession(
            spec=self.live_session.machine_spec,
            center=self.live_session.machine_center,
            angle=self.live_session.machine_angle,
            routing_region=self.live_session.routing_region,
            columns=self.columns,
            workspace=self.live_session.workspace,
            graph_type=self.graph_type,
            update_dynamic_env=self.callbacks.update_dynamic_env,
            build_grid=self.callbacks.build_grid,
            blocked_vertical_regions=self.blocked_vertical_regions,
        )
        analysis = self.live_session.route_analysis(self.shaft_extraction)
        route_plan = self.live_session.installation.build_route_plan(
            self.terminals, self.live_session.machine_center,
        )

        def materializer():
            return self.route_materializer_factory(route_plan)

        callbacks = SalLiveRoutingCallbacks(
            shaft_entry_nodes=self.callbacks.shaft_entry_nodes,
            terminal_nodes=self.callbacks.terminal_nodes,
            block_terminal_edges=self.callbacks.block_terminal_edges,
            source_start_nodes=lambda source: materializer().source_start_nodes(source),
            weighted_edge_cost=self.callbacks.weighted_edge_cost,
            line_graph_direction=self.callbacks.line_graph_direction,
            route_start_nodes=self.callbacks.route_start_nodes,
            route_segments_from_path=lambda _plan, *args: materializer().route_segments(*args),
            build_routes_from_paths=lambda _plan, *args: materializer().build_routes(*args),
            count_crossings=self.callbacks.count_crossings,
            score_routes=lambda routes, crossings, _policy: analysis.score(routes, crossings),
            conflict_summary=analysis.conflict_summary,
        )

        workspace = self.live_session.workspace
        workspace.overlay.reset()
        result = self.live_session.application_adapter(
            machine_session, self.terminals, callbacks,
        ).solve()
        workspace.overlay.excluded_edges.update(result.excluded_overlay_edges)
        return result
