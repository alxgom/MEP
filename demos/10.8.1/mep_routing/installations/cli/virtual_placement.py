"""Rank valid CLI machine poses with one supply-tree search."""

from dataclasses import dataclass, replace

import numpy as np

from mep_routing.app.dwelling_application import InstallationGraphProfile, create_active_dwelling
from mep_routing.geometry.directions import travel_direction
from mep_routing.graphs import GraphMode, build_simple_grid
from mep_routing.graphs.runtime import build_adjacency

from .definition import CLI_FITTING_BACKEND
from .directed_wall_graph import _prepare_wall_graph
from .fittings import FITTING_PERTURBATION_MM
from .geometry import (
    cli_clear_supply_turns, cli_machine_footprint, cli_machine_pins, cli_supply_access_specs,
    cli_supply_exit_footprints,
)
from .graph_inputs import build_cli_simple_graph_inputs
from .length_steiner import TreeLengthPolicy, rank_length_steiner_roots
from .placement import CliPlacementResult


@dataclass(frozen=True)
class CliVirtualPlacementResult:
    placement: CliPlacementResult | None
    status: str
    feasible: int
    ranked: int
    verified: int
    elapsed_ms: float


class CliVirtualPlacementSearch:
    """Reuse the static hybrid grid as the source of placement centers."""

    graph_mode = GraphMode.CORE_SIMPLE
    backend_index = CLI_FITTING_BACKEND

    def __init__(self, graph_config, terminal_config):
        self.graph_config = graph_config
        self.terminal_config = terminal_config
        self._dwelling = None
        self._grid = None
        self._centers = ()
        self._mask = None
        self._screened = ()

    def candidate_grid(self, dwelling):
        if dwelling is not self._dwelling:
            static = replace(dwelling, terminals={}, wet_room_names=[])
            profile = InstallationGraphProfile(lambda _pins, _angle: ())
            workspace = create_active_dwelling(
                static, self.graph_config, profile, self.terminal_config,
            ).workspace
            grid = workspace.build_selected(GraphMode.HANNAN_CELL, {}, 0)
            self._grid = grid.variant if grid is not None else None
            self._centers = tuple(sorted({
                (round(float(x)), round(float(y))) for x, y in self._grid.nodes
            })) if self._grid is not None else ()
            self._dwelling = dwelling
            self._mask = None
        return self._grid

    def screened_centers(self, dwelling, mask):
        self.candidate_grid(dwelling)
        if mask is not self._mask:
            self._screened = tuple(
                (point, status) for point in self._centers
                if (status := mask.center_status(point)) != "blocked"
            )
            self._mask = mask
        return self._screened

    def rank(self, dwelling, machine, angle, mask, grilles, exit_dimensions, minimum_length_mm,
             bend_penalty_mm, context_obstacles):
        length, width = exit_dimensions
        obstacles = (*dwelling.columns, *dwelling.shafts, *context_obstacles)
        candidates = []
        for center, _status in self.screened_centers(dwelling, mask):
            pins = cli_machine_pins(machine, center, angle)
            body = cli_machine_footprint(pins)
            if not dwelling.routing_region_base.covers(body) or any(body.intersects(o) for o in obstacles):
                continue
            access, = cli_supply_access_specs(machine, pins, angle, length)
            extension, turns = cli_supply_exit_footprints(machine, angle, access, width)
            clear = cli_clear_supply_turns(dwelling, extension, turns, context_obstacles)
            if clear:
                blocked = {access["in_dir"], *(turns.keys() - clear.keys())}
                candidates.append((center, access, blocked))
        if not candidates:
            return (), 0

        terminals = {
            (name, i): supply.routing_point
            for name, pairs in grilles.pairs_by_supply.items()
            for i, (supply, _) in enumerate(pairs)
        }
        inputs = build_cli_simple_graph_inputs(
            routing_region=dwelling.routing_region_base,
            terminals=terminals, columns=dwelling.columns, shafts=dwelling.shafts,
            machine_pins={}, access_points=(access["access_point"] for _, access, _ in candidates),
            epsilon_mm=self.graph_config.epsilon_mm, axis_clearance_mm=100,
            context_obstacles=context_obstacles,
        )
        grid = build_simple_grid(inputs)
        adjacency = build_adjacency(len(grid.nodes), (
            edge for edge in grid.edges if edge[2] < inputs.blocked_edge_weight
        ))
        prepared = _prepare_wall_graph(grid.nodes, adjacency, grilles.pairs_by_supply, grilles.supply)
        if prepared is None:
            return (), len(candidates)

        node_at = {tuple(point): index for index, point in enumerate(grid.nodes)}
        terminals_at = {}
        for (name, node), wall in prepared.wall_points.items():
            terminals_at.setdefault(node, []).append((name, wall))
        terminal_nodes = {name: list(nodes) for name, nodes in prepared.candidates.items()}
        wall_points = dict(prepared.wall_points)
        roots, placements, extra_points = [], {}, []
        for center, access, blocked in candidates:
            point = access["access_point"]
            node = node_at[point]
            edges = [edge for edge in adjacency[node] if edge[2] not in blocked]
            direct = [(name, wall) for name, wall in terminals_at.get(node, ())
                      if travel_direction(point, wall) not in blocked]
            if not edges and not direct:
                continue
            # Each outlet has an entry node with no incoming arcs. Only the
            # chosen outlet's first directions constrain its shared tree.
            root = len(grid.nodes) + len(extra_points)
            extra_points.append(point)
            adjacency[root] = edges
            for name, wall in direct:
                terminal_nodes[name].append(root)
                wall_points[(name, root)] = wall
            roots.append((root, ((access["pin_point"], point),)))
            placements[root] = center
        if not roots:
            return (), len(candidates)
        prepared = replace(prepared, nodes=np.vstack((grid.nodes, extra_points)),
                           adjacency=adjacency, candidates=terminal_nodes, wall_points=wall_points)
        scores = rank_length_steiner_roots(
            prepared, roots, tuple(prepared.candidates), bend_penalty_mm,
            TreeLengthPolicy(minimum_length_mm, split_at_tees=True),
            fitting_weight=FITTING_PERTURBATION_MM,
        )
        return tuple(placements[root] for root, _ in scores), len(candidates)
