"""Bound machine-placement state for the interactive application."""

from __future__ import annotations

from dataclasses import dataclass, field

from .application import PlacementApplicationAdapter
from .feasibility import is_machine_placement_valid
from .fields import placement_weights
from .vertical_clearance import available_machine_placement_region, scores_outside_regions


@dataclass
class PlacementSession:
    """Pair placement policy with the currently active dwelling and graph."""

    machine_area: float
    machine_pins: object
    representative_point: object
    route_room_polygon: object
    local_axis_to_world: object
    graph_builder: object
    shaft_nodes: object
    weight_mode: int = 0
    rotation_eps: float = 1e-6
    workspace: object = None
    rooms: tuple = ()
    terminals: dict = field(default_factory=dict)
    wet_room_names: tuple = ()
    routing_region: object = None
    shaft: object = None
    clearance_blocks: tuple = ()
    walls: tuple = ()
    columns: tuple = ()
    shafts: tuple = ()
    scores: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)
    rotation_scores: dict = field(default_factory=lambda: {"H": 0.0, "V": 0.0, "selected": None})

    def bind(self, workspace, rooms, terminals, wet_room_names, routing_region, shaft,
             clearance_blocks, walls=(), columns=(), shafts=()):
        self.workspace = workspace
        self.rooms = tuple(rooms)
        self.terminals = dict(terminals)
        self.wet_room_names = tuple(wet_room_names)
        self.routing_region = routing_region
        self.shaft = shaft
        self.clearance_blocks = tuple(clearance_blocks)
        self.walls, self.columns, self.shafts = tuple(walls), tuple(columns), tuple(shafts)
        self.scores, self.fields = {}, {}

    @property
    def placeable_region(self):
        return available_machine_placement_region(self.routing_region, self.clearance_blocks)

    @property
    def weights(self):
        return placement_weights(self.weight_mode)

    def is_valid(self, cx, cy, angle):
        return is_machine_placement_valid(
            cx, cy, self.machine_pins(cx, cy, angle), self.routing_region,
            self.walls, self.columns, self.shafts, self.clearance_blocks,
            tuple(room.polygon for room in self.rooms if hasattr(room, "polygon")),
        )

    def _adapter(self):
        # The full feasibility callback is injected by bind_client because Demo
        # wall/column/shaft collections remain application compatibility state.
        return PlacementApplicationAdapter(
            rooms=self.rooms, terminals=self.terminals, wet_room_names=self.wet_room_names,
            routing_region=self.routing_region, shaft=self.shaft,
            machine_area=self.machine_area, weight_mode=self.weight_mode, weights=self.weights,
            machine_pins=self.machine_pins, is_valid=self.is_valid,
            representative_point=self.representative_point,
            route_room_polygon=self.route_room_polygon, local_axis_to_world=self.local_axis_to_world,
            placeable_region=self.placeable_region,
        )

    def placement_scores(self, env, spatial_index, shaft_boundary_nodes):
        return self._adapter().topological_scores(env, spatial_index, shaft_boundary_nodes)

    def ensure_scores(self):
        if self.scores or self.workspace is None or self.workspace.base_env is None or self.shaft is None:
            return
        boundary_nodes, _ = self.shaft_nodes(
            self.workspace.base_env, self.workspace.base_spatial_index,
        )
        self.scores, self.fields = self.placement_scores(
            self.workspace.base_env, self.workspace.base_spatial_index, boundary_nodes,
        )

    def align_rotation(self, center, angle):
        angle, self.rotation_scores, raw_scores = self._adapter().align_rotation(
            center, angle, self.rotation_eps,
        )
        return angle, self.rotation_scores["selected"], raw_scores

    def apply_rotation_mode(self, mode, center, angle):
        if mode == 0:
            self.rotation_scores.update({"H": 0.0, "V": 0.0, "selected": "Torque"})
            return angle
        new_angle = self.align_rotation(center, angle)[0]
        if new_angle != angle:
            self.graph_builder(machine_pins=self.machine_pins(center[0], center[1], new_angle))
        return new_angle

    def auto_place(self, mode):
        if self.workspace is None or self.workspace.base_env is None or self.shaft is None:
            return None, 0.0
        boundary_nodes, _ = self.shaft_nodes(
            self.workspace.base_env, self.workspace.base_spatial_index,
        )
        outcome, elapsed_ms = self._adapter().auto_place(
            mode, self.workspace.base_env, self.workspace.base_spatial_index, boundary_nodes,
        )
        if outcome is not None:
            self.scores, self.fields = outcome.scores, outcome.fields
        return outcome, elapsed_ms

    def placeable_scores(self, scores):
        if self.workspace is None or self.workspace.base_env is None:
            return {}
        return scores_outside_regions(scores, self.workspace.base_env.nodes, self.clearance_blocks)
