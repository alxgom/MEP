"""Two-dimensional Coc routing between exported appliance and shaft terminals."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from mep_routing.app.dwelling_application import InstallationGraphProfile, create_active_dwelling
from mep_routing.graphs import GraphMode, adjacency_without_obstacle_bounds
from mep_routing.routing import segments_from_edges, solve_steiner_tree

from .graph_inputs import build_coc_simple_graph_inputs


@dataclass(frozen=True)
class CocRoutingResult:
    routes: tuple[tuple[str, list], ...]
    status: str
    elapsed_ms: float
    total_nodes: int
    terminals: dict[str, tuple[float, float]]
    machine_pins: dict[str, tuple[float, float]]
    workspace: object | None = None
    shaft_extraction: object | None = None
    diameter_mm: int = 125
    context_obstacle_count: int = 0


def _appliance_pins(appliance):
    polygon = appliance.get("polygon")
    exhaust = appliance.get("exhaust_xy")
    pins = {}
    if polygon is not None and not polygon.is_empty:
        min_x, min_y, max_x, max_y = polygon.bounds
        pins.update({
            "c_tl": (min_x, max_y), "c_tr": (max_x, max_y),
            "c_br": (max_x, min_y), "c_bl": (min_x, min_y),
        })
    if exhaust:
        pins["exhaust"] = tuple(map(float, exhaust))
    return pins


class CocInteractiveRuntime:
    """Route Coc in plan only; downstream code owns height and 3D placement."""

    def __init__(self, definition, graph_config, terminal_config):
        self.definition = definition
        self.graph_config = graph_config
        self.terminal_config = terminal_config
        self.last_result = CocRoutingResult((), "Coc has not run", 0.0, 0, {}, {})

    def solve(self, prepared, *, context_obstacles=(), graph_mode=None):
        started = perf_counter()
        payload = (prepared.installation_inputs or {}).get("coc") or {}
        appliance = payload.get("appliance")
        shaft_entries = payload.get("shaft_entries") or ()
        if not appliance or appliance.get("source_status") != "ready" or not appliance.get("exhaust_xy"):
            return self._finish((), "Blocked: Coc source unavailable", started, 0, {}, {})
        if not shaft_entries:
            return self._finish((), "Blocked: Coc shaft entry unavailable", started, 0, {}, _appliance_pins(appliance))

        source = tuple(map(float, appliance["exhaust_xy"]))
        target = tuple(map(float, shaft_entries[0]["entry"]))
        normal = tuple(map(float, shaft_entries[0]["normal"]))
        shaft_hint = (
            target[0] + normal[0] * self.graph_config.wall_clearance_mm * 2.0,
            target[1] + normal[1] * self.graph_config.wall_clearance_mm * 2.0,
        )
        source_access = source
        shaft_access = shaft_hint
        graph_terminals = {"Coc source": source_access, "Coc shaft": shaft_access}
        terminals = {"Coc source": source, "Coc shaft": target}

        def simple_inputs(**kwargs):
            return build_coc_simple_graph_inputs(**kwargs)

        routed = replace(
            prepared, terminals=graph_terminals, wet_room_names=[],
            shaft_extraction=None, shaft_core_entry_specs=[],
        )
        profile = InstallationGraphProfile(lambda _pins, _angle: (), simple_inputs)
        dwelling = create_active_dwelling(
            routed, self.graph_config, profile, self.terminal_config,
        )
        mode = self.definition.graph_modes[0] if graph_mode is None else graph_mode
        build = dwelling.workspace.build_selected(mode, {}, 0.0)
        if build is None or not len(build.runtime.nodes):
            return self._finish((), "Blocked: Coc graph unavailable", started, 0, terminals, _appliance_pins(appliance))
        runtime = build.runtime
        indices = tuple(int(runtime.spatial_index.query(point)[1]) for point in (source_access, shaft_access))
        context_bounds = tuple(obstacle.bounds for obstacle in context_obstacles)
        adjacency, _blocked_nodes, _blocked_edges = adjacency_without_obstacle_bounds(
            runtime, context_bounds, indices,
        )
        tree = solve_steiner_tree(runtime.nodes, adjacency, indices)
        used_regular_fallback = False
        if tree is None and GraphMode(mode) is GraphMode.CORE_SIMPLE:
            fallback = dwelling.workspace.build_selected(GraphMode.REGULAR, {}, 0.0)
            if fallback is not None and len(fallback.runtime.nodes):
                runtime = fallback.runtime
                indices = tuple(
                    int(runtime.spatial_index.query(point)[1])
                    for point in (source_access, shaft_access)
                )
                adjacency, _blocked_nodes, _blocked_edges = adjacency_without_obstacle_bounds(
                    runtime, context_bounds, indices,
                )
                tree = solve_steiner_tree(runtime.nodes, adjacency, indices)
                used_regular_fallback = tree is not None
        if tree is None:
            return self._finish(
                (), "Blocked: no Coc exhaust path", started, len(runtime.nodes),
                terminals, _appliance_pins(appliance), workspace=dwelling.workspace,
            )
        segments = list(segments_from_edges(runtime.nodes, tree.tree_edges))
        graph_source = tuple(map(float, runtime.nodes[indices[0]][:2]))
        graph_target = tuple(map(float, runtime.nodes[indices[1]][:2]))
        if source != graph_source:
            segments.insert(0, (source, graph_source))
        if graph_target != target:
            segments.append((graph_target, target))
        diameter_mm = round(float((payload.get("duct") or {}).get("diameter_m", 0.125)) * 1000)
        return self._finish(
            (("Kitchen Exhaust", segments),),
            "Success: Coc exhaust solved" + (" (regular-grid fallback)" if used_regular_fallback else ""),
            started,
            len(runtime.nodes), terminals, _appliance_pins(appliance),
            workspace=dwelling.workspace, diameter_mm=diameter_mm,
            shaft_extraction=payload.get("shaft"),
            context_obstacles=context_obstacles,
        )

    def _finish(
        self, routes, status, started, total_nodes, terminals, pins, *, workspace=None,
        shaft_extraction=None, diameter_mm=125, context_obstacles=(),
    ):
        self.last_result = CocRoutingResult(
            tuple(routes), status, (perf_counter() - started) * 1000.0,
            total_nodes, dict(terminals), dict(pins), workspace,
            shaft_extraction, diameter_mm, len(tuple(context_obstacles)),
        )
        return self.last_result
