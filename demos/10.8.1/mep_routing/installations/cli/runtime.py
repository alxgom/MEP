"""Cli-owned interactive supply and return-air runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from time import perf_counter

from shapely.geometry import Polygon

from mep_routing.app import InstallationGraphProfile, create_active_dwelling
from mep_routing.graphs import GraphMode

from .geometry import cli_air_access_spec, cli_machine_pins
from .definition import (
    CLI_DIRECTED_BACKENDS, CLI_FITTING_BACKEND, CLI_LENGTH_BACKENDS, CLI_TEE_LENGTH_BACKENDS,
)
from .fittings import FITTING_PERTURBATION_MM
from .clearance import apply_cli_edge_weights, cli_minimum_clearance_weights
from .duct_geometry import build_core_duct_polygon_projection
from .directed_wall_graph import solve_directed_wall_steiner, solve_directed_wall_tree
from .aligned_duct_geometry import (
    CliDuctAlignmentConstraints, build_aligned_duct_polygon_projection,
)
from .graph_inputs import build_cli_simple_graph_inputs
from .grilles import CliGrillePlan, prepare_cli_grilles
from .materializer import CliDuctSegment, materialize_cli_tree
from .length_steiner import LengthSteinerStats, TreeLengthPolicy, validate_tree_lengths
from .placement import CliPlacementResult, CliPlacementRuntime
from .solver_backends import solve_cli_tree
from .sizing import (
    CliSizingContext, CoreConstantFrictionSizing, CoreSupplyGrilleSizing,
    size_tree_edges,
)
from .terminals import is_cli_supply_room


@dataclass(frozen=True)
class CliRoutingResult:
    routes: tuple[tuple[str, list], ...]
    status: str
    elapsed_ms: float
    total_nodes: int
    terminals: dict[str, tuple[int, int]]
    machine_pins: dict[str, tuple[int, int]]
    workspace: object | None = None
    grille_plan: CliGrillePlan | None = None
    supply_segments: tuple[CliDuctSegment, ...] = ()
    return_segments: tuple[CliDuctSegment, ...] = ()
    placement: CliPlacementResult | None = None
    context_obstacle_count: int = 0
    context_obstacles: tuple = ()
    supply_minimum_length_mm: float | None = None
    supply_length_stats: LengthSteinerStats | None = None


@dataclass(frozen=True)
class _SolvedTree:
    workspace: object
    segments: tuple[CliDuctSegment, ...]
    node_count: int
    connections: tuple = ()
    minimum_length_mm: float | None = None
    length_stats: LengthSteinerStats | None = None


class CliInteractiveRuntime:
    """Compose Cli capabilities without consuming Sal placement or solver contracts."""

    def __init__(
        self, definition, graph_config, terminal_config, placement=None,
        sizing_policy_factory=CoreConstantFrictionSizing,
        grille_sizing_policy_factory=CoreSupplyGrilleSizing,
    ):
        self.definition = definition
        self.graph_config = graph_config
        self.terminal_config = terminal_config
        self.placement = placement or CliPlacementRuntime()
        self.sizing_policy_factory = sizing_policy_factory
        self.grille_sizing_policy_factory = grille_sizing_policy_factory
        self._core_polygon_result = None
        self._core_polygon_projection = None
        self._aligned_polygon_result = None
        self._aligned_polygon_projection = None
        self.last_placement = None
        self.last_result = CliRoutingResult((), "Cli has not run", 0.0, 0, {}, {})

    def place_machine(self, prepared_dwelling, machine, *, context_obstacles=()):
        grille_sizes = self._size_supply_grilles(prepared_dwelling.rooms, machine)
        grilles = prepare_cli_grilles(
            prepared_dwelling.rooms, prepared_dwelling.routing_region_base, machine=machine,
            supply_width_by_room_id=grille_sizes,
        )
        self.last_placement = self.placement.place(
            prepared_dwelling, machine, grilles, context_obstacles=context_obstacles,
        )
        return self.last_placement

    def solve(
        self, prepared_dwelling, machine, center, angle_degrees=0, *,
        context_obstacles=(), return_enabled=True, graph_mode=GraphMode.CORE_SIMPLE,
        solver_backend_index=0, bend_penalty_mm=4000.0,
        crossing_penalty_multiplier=5.0,
        start_mode_index=0,
        minimum_segment_length_factor=1.0,
    ):
        started = perf_counter()
        self._length_failure = None
        self._length_stats = None
        self._length_minimum = None
        pins = cli_machine_pins(machine, center, angle_degrees)
        grille_sizes = self._size_supply_grilles(prepared_dwelling.rooms, machine)
        grilles = prepare_cli_grilles(
            prepared_dwelling.rooms, prepared_dwelling.routing_region_base,
            machine=machine, machine_pins=pins, angle_degrees=angle_degrees,
            supply_width_by_room_id=grille_sizes,
        )
        if not return_enabled:
            grilles = CliGrillePlan(tuple(grilles.supply))
        if not grilles.supply:
            return self._finish(
                (), "Blocked: no Cli supply grilles", started, 0, grilles, pins,
                context_obstacles=context_obstacles,
            )

        if solver_backend_index in CLI_LENGTH_BACKENDS:
            if not math.isfinite(minimum_segment_length_factor) or minimum_segment_length_factor <= 0:
                raise ValueError("minimum length factor must be finite and positive")
            reason = None
            if start_mode_index != 1:
                reason = "choose Wall candidates for hard Steiner lengths"
            elif not (machine.airflow_m3h and machine.pressure_loss_mmca):
                reason = "selected machine has no calculated duct width"
            if reason:
                return self._finish(
                    (), f"Unsupported: {reason}", started, 0, grilles, pins,
                    context_obstacles=context_obstacles,
                )

        supply = self._solve_tree(
            "supply", prepared_dwelling, machine, pins, angle_degrees,
            grilles, context_obstacles, graph_mode, solver_backend_index,
            bend_penalty_mm, crossing_penalty_multiplier,
            start_mode_index=start_mode_index,
            minimum_segment_length_factor=minimum_segment_length_factor,
        )
        if supply is None:
            if context_obstacles:
                relaxed = self.solve(
                    prepared_dwelling, machine, center, angle_degrees,
                    return_enabled=return_enabled, graph_mode=graph_mode,
                    solver_backend_index=solver_backend_index,
                    bend_penalty_mm=bend_penalty_mm,
                    crossing_penalty_multiplier=crossing_penalty_multiplier,
                    start_mode_index=start_mode_index,
                    minimum_segment_length_factor=minimum_segment_length_factor,
                )
                if relaxed.routes:
                    self.last_result = replace(
                        relaxed,
                        status=f"Cli context relaxed after collision; {relaxed.status}",
                        context_obstacle_count=len(tuple(context_obstacles)),
                        context_obstacles=tuple(context_obstacles),
                    )
                    return self.last_result
            return self._finish(
                (), self._length_failure or (
                    "Blocked: hard-length search found no supply tree" if solver_backend_index in CLI_LENGTH_BACKENDS
                    else "Blocked: no Cli supply tree"
                ), started, 0, grilles, pins,
                context_obstacles=context_obstacles,
            )
        returned = None
        if return_enabled:
            returned = self._solve_tree(
                "return", prepared_dwelling, machine, pins, angle_degrees,
                grilles, context_obstacles, graph_mode, solver_backend_index,
                bend_penalty_mm, crossing_penalty_multiplier,
                start_mode_index=0,
                prior_segments=tuple(segment.line for segment in supply.segments),
            )
        if supply.connections:
            selected = {item.route_name: item for item in supply.connections}
            grilles = CliGrillePlan(tuple(
                selected.get(item.route_name, item) for item in grilles.connections
            ))
        routes = (("Supply Air Tree", [segment.line for segment in supply.segments]),)
        status = f"Cli supply solved: {len(grilles.supply)} grilles"
        return_segments = ()
        total_nodes = supply.node_count
        if returned is not None:
            return_segments = returned.segments
            routes += (("Return Air Tree", [segment.line for segment in returned.segments]),)
            total_nodes += returned.node_count
            status = (
                f"Cli supply/return solved: {len(grilles.supply)} rooms, "
                f"{len(supply.segments) + len(return_segments)} segments"
            )
        elif return_enabled:
            status += "; return tree blocked"
        if supply.minimum_length_mm is not None:
            mode = "bends + tees" if solver_backend_index in CLI_TEE_LENGTH_BACKENDS else "bends"
            status += f"; hard {mode} min {supply.minimum_length_mm:g} mm"
            if solver_backend_index == CLI_FITTING_BACKEND:
                status += f"; fittings {supply.length_stats.fittings.total}, eps {FITTING_PERTURBATION_MM:g}"
        return self._finish(
            routes, status, started, total_nodes, grilles, pins,
            workspace=supply.workspace, supply_segments=supply.segments,
            return_segments=return_segments, context_obstacles=context_obstacles,
        )

    def _size_supply_grilles(self, rooms, machine):
        if not machine.airflow_m3h:
            return {}
        targets = tuple(room for room in rooms if is_cli_supply_room(room))
        areas = {id(room): float(room.polygon.area) / 1_000_000.0 for room in targets}
        total_area = sum(areas.values())
        if total_area <= 0:
            return {}
        policy = self.grille_sizing_policy_factory(machine.airflow_m3h)
        return {
            id(room): policy.size(areas[id(room)], total_area)
            for room in targets
        }

    def _solve_tree(
        self, role, prepared, machine, pins, angle, grilles, context_obstacles,
        graph_mode, solver_backend_index, bend_penalty_mm,
        crossing_penalty_multiplier, prior_segments=(), start_mode_index=0,
        minimum_segment_length_factor=1.0,
    ):
        connections = grilles.supply if role == "supply" else grilles.returns
        if not connections:
            return None
        connector_key = "air_out" if role == "supply" else "air_in"
        policy, machine_sizing, grille_sizes = None, None, {}
        area_by_route = {item.route_name: item.room_area_m2 for item in connections}
        if role == "supply" and machine.airflow_m3h and machine.pressure_loss_mmca:
            policy = self.sizing_policy_factory(CliSizingContext(
                machine.airflow_m3h, machine.pressure_loss_mmca,
            ))
            total_area = float(sum(area_by_route.values()))
            if total_area > 0:
                machine_sizing = policy.size(total_area, total_area)
                grille_sizes = {
                    item.route_name: policy.size(item.room_area_m2, total_area)
                    for item in connections
                }
        access_length_mm = 500 if role == "supply" else 20
        access = cli_air_access_spec(
            machine, pins, angle, connector_key, access_length_mm,
        )
        terminals = {item.route_name: item.routing_point for item in connections}
        role_prepared = replace(prepared, terminals=terminals, wet_room_names=[])

        def simple_inputs(**kwargs):
            return build_cli_simple_graph_inputs(
                **kwargs, context_obstacles=context_obstacles,
            )

        profile = InstallationGraphProfile(
            port_access_specs=lambda current_pins, current_angle: (
                cli_air_access_spec(
                    machine, current_pins, current_angle, connector_key,
                    access_length_mm,
                ),
            ),
            simple_input_builder=simple_inputs,
        )
        dwelling = create_active_dwelling(
            role_prepared, self.graph_config, profile, self.terminal_config,
        )
        build = dwelling.workspace.build_selected(GraphMode(graph_mode), pins, angle)
        if build is None or not len(build.runtime.nodes):
            return None
        runtime = build.runtime
        corner_keys = ("c_tl", "c_tr", "c_br", "c_bl")
        machine_center = (
            sum(pins[key][0] for key in corner_keys) / len(corner_keys),
            sum(pins[key][1] for key in corner_keys) / len(corner_keys),
        )
        clearance_weights = cli_minimum_clearance_weights(
            prepared, machine, machine_center, angle, runtime,
        )
        weighted_adjacency = apply_cli_edge_weights(runtime.adjacency, clearance_weights)
        dwelling.workspace.overlay.values = dict(clearance_weights)
        root_node = int(runtime.spatial_index.query(access["access_point"])[1])
        graph_nodes, graph_adjacency = runtime.nodes, weighted_adjacency
        selected_connections = tuple(connections)
        length_policy, length_stats = None, None
        if role == "supply" and solver_backend_index in CLI_LENGTH_BACKENDS:
            if machine_sizing is None:
                self._length_failure = "Unsupported: no calculated full-flow duct width"
                return None
            self._length_minimum = machine_sizing.width_mm * minimum_segment_length_factor
            length_policy = TreeLengthPolicy(self._length_minimum, solver_backend_index in CLI_TEE_LENGTH_BACKENDS)
            length_stats = self._length_stats = LengthSteinerStats()
            if math.dist(runtime.nodes[root_node, :2], access["access_point"]) > 1e-7:
                self._length_failure = "Unsupported: graph snaps the machine access; use Hannan or Core Simple"
                return None
        if role == "supply" and start_mode_index == 1 and solver_backend_index in CLI_DIRECTED_BACKENDS:
            targets = [room for room in prepared.rooms if is_cli_supply_room(room)]
            rooms_by_name = {
                item.room_name: room for item, room in zip(connections, targets)
            }
            directed_solver = (
                solve_directed_wall_steiner if solver_backend_index != 3
                else solve_directed_wall_tree
            )
            extra = {} if length_policy is None else dict(
                length_policy=length_policy,
                root_prefix=((pins[connector_key], access["access_point"]),),
                stats=length_stats,
                fitting_weight=FITTING_PERTURBATION_MM if solver_backend_index == CLI_FITTING_BACKEND else 0.0,
            )
            directed = directed_solver(
                runtime.nodes, weighted_adjacency, root_node, rooms_by_name,
                connections, bend_penalty_mm, **extra,
            )
            if directed is None:
                if length_stats is not None and length_stats.rejected_physical_tree:
                    self._length_failure = "Blocked: constrained candidate failed physical-tree check"
                return None
            tree, graph_nodes, graph_adjacency = directed.tree, directed.nodes, directed.adjacency
            selected_connections = directed.selected_connections
            terminals = {item.route_name: item.routing_point for item in selected_connections}
        else:
            points = [access["access_point"], *terminals.values()]
            indices = tuple(dict.fromkeys(int(runtime.spatial_index.query(point)[1]) for point in points))
            effective_backend = 2 if solver_backend_index in CLI_DIRECTED_BACKENDS else solver_backend_index
            tree = solve_cli_tree(
                effective_backend, runtime.nodes, weighted_adjacency, indices,
                bend_penalty=bend_penalty_mm,
                crossing_multiplier=crossing_penalty_multiplier,
                prior_segments=prior_segments,
            )
        if tree is None:
            return None
        edge_sizes = {}
        if policy is not None:
            area_by_node = {}
            for name, point in terminals.items():
                distances = ((graph_nodes[:, :2] - point) ** 2).sum(axis=1)
                node = int(distances.argmin())
                area_by_node[node] = area_by_node.get(node, 0.0) + area_by_route.get(name, 0.0)
            edge_sizes = size_tree_edges(tree.tree_edges, root_node, area_by_node, policy)
        segments = materialize_cli_tree(
            role=role, machine=machine, pins=pins, access_spec=access,
            graph_nodes=graph_nodes, tree=tree, grille_connections=selected_connections,
            edge_sizes=edge_sizes, machine_sizing=machine_sizing,
            grille_sizes=grille_sizes,
        )
        if length_policy is not None and not validate_tree_lengths(
            [item.line for item in segments], pins[connector_key],
            [item.grille_point for item in selected_connections], length_policy,
        ):
            length_stats.rejected_physical_tree = True
            self._length_failure = "Blocked: materialized tree failed hard length check"
            return None
        return _SolvedTree(
            dwelling.workspace, segments, len(graph_nodes), selected_connections,
            length_policy.minimum_mm if length_policy is not None else None, length_stats,
        )

    def _finish(
        self, routes, status, started, total_nodes, grilles, pins, *, workspace=None,
        supply_segments=(), return_segments=(), context_obstacles=(), placement=None,
    ):
        context_obstacles = tuple(context_obstacles)
        self.last_result = CliRoutingResult(
            tuple(routes), status, (perf_counter() - started) * 1000.0, total_nodes,
            grilles.grille_terminals(), dict(pins), workspace, grilles,
            tuple(supply_segments), tuple(return_segments), placement or self.last_placement,
            len(context_obstacles), context_obstacles,
            self._length_minimum, self._length_stats,
        )
        self._core_polygon_result = None
        self._core_polygon_projection = None
        self._aligned_polygon_result = None
        self._aligned_polygon_projection = None
        return self.last_result

    def core_polygon_projection(self, result=None):
        """Build once per routing result, then reuse across render frames."""
        result = self.last_result if result is None else result
        if result is not self._core_polygon_result:
            self._core_polygon_projection = build_core_duct_polygon_projection(
                result.supply_segments,
            )
            self._core_polygon_result = result
        return self._core_polygon_projection

    def aligned_polygon_projection(self, result=None):
        """Build the experimental projection once per routing result."""
        result = self.last_result if result is None else result
        if result is not self._aligned_polygon_result:
            self._aligned_polygon_projection = build_aligned_duct_polygon_projection(
                result.supply_segments, self._alignment_constraints(result),
            )
            self._aligned_polygon_result = result
        return self._aligned_polygon_projection

    @staticmethod
    def _alignment_constraints(result):
        workspace = getattr(result, "workspace", None)
        lifecycle = getattr(workspace, "lifecycle", None)
        if lifecycle is None:
            return None
        obstacles = []
        for name in ("wall_polygons", "columns", "shafts"):
            obstacles.extend(tuple(getattr(lifecycle, name, ()) or ()))
        obstacles.extend(tuple(getattr(result, "context_obstacles", ()) or ()))
        pins = getattr(result, "machine_pins", {}) or {}
        corners = [pins.get(name) for name in ("c_tl", "c_tr", "c_br", "c_bl")]
        if all(point is not None for point in corners):
            obstacles.append(Polygon(corners))
        return CliDuctAlignmentConstraints(
            getattr(lifecycle, "routing_region", None), tuple(obstacles),
        )
