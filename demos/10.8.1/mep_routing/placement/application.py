"""Application boundary for machine placement workflows."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from shapely.geometry import Point

from .feasibility import candidate_machine_rooms
from .fields import topological_placement_scores
from .rotation import score_rotation_field_at, select_field_alignment_rotation
from .runtime import PlacementOutcome, run_core_like_placement, run_topological_placement
from .scoring import candidate_room_points, core_like_machine_candidate_score, routing_frame_axes
from .selection import choose_core_like_machine_placement, choose_topological_machine_placement


@dataclass(frozen=True)
class PlacementApplicationAdapter:
    """Snapshot of geometry and callbacks needed to evaluate machine placement."""

    rooms: tuple
    terminals: dict
    wet_room_names: tuple
    routing_region: object
    shaft: object
    machine_area: float
    weight_mode: int
    weights: dict
    machine_pins: object
    is_valid: object
    representative_point: object
    route_room_polygon: object
    local_axis_to_world: object
    rotations: tuple = (0, 90, 180, 270)

    def topological_scores(self, env, spatial_index, shaft_nodes):
        terminal_nodes = {
            name: int(spatial_index.query(point)[1])
            for name, point in self.terminals.items()
        }
        return topological_placement_scores(env, shaft_nodes, terminal_nodes, self.weights)

    def auto_place(self, mode, env, spatial_index, shaft_nodes):
        started = time.perf_counter()
        if mode == 1:
            outcome = run_topological_placement(
                env,
                shaft_nodes,
                self.rotations,
                self.is_valid,
                self.machine_pins,
                lambda point: int(spatial_index.query(point)[1]),
                self.wet_room_names,
                self.weights,
                lambda current_env, current_shaft_nodes: self.topological_scores(
                    current_env, spatial_index, current_shaft_nodes,
                ),
                choose_topological_machine_placement,
            )
        elif mode == 2:
            outcome = run_core_like_placement(
                candidate_machine_rooms(self.rooms, self.machine_area),
                lambda room: candidate_room_points(room, routing_frame_axes()),
                self.rotations,
                self.is_valid,
                self._core_candidate_score,
                choose_core_like_machine_placement,
            )
        else:
            return None, 0.0
        return outcome, (time.perf_counter() - started) * 1000.0

    def align_rotation(self, center, current_angle, eps):
        cx, cy = center
        angle, selected, scores = select_field_alignment_rotation(
            current_angle,
            lambda candidate: self.is_valid(cx, cy, candidate),
            lambda candidate: self._rotation_score(cx, cy, candidate),
            eps,
        )
        display_scores = {
            key: 0.0 if not math.isfinite(scores.get(key, 0.0)) else float(scores[key])
            for key in ("H", "V")
        }
        display_scores["selected"] = selected
        return angle, display_scores, scores

    def _boundary_distance(self, point):
        if self.routing_region is None:
            return 1e9
        return Point(float(point[0]), float(point[1])).distance(self.routing_region.boundary)

    def _core_candidate_score(self, cx, cy, angle, room):
        shaft_point = self.representative_point(self.shaft) if self.shaft else (cx, cy)
        kitchen_point = self.terminals.get("Kitchen", (cx, cy))
        return core_like_machine_candidate_score(
            cx,
            cy,
            angle,
            room.polygon,
            self.machine_pins(cx, cy, angle),
            shaft_point,
            kitchen_point,
            "Kitchen" in self.terminals,
            self._boundary_distance,
            self.local_axis_to_world,
        )

    def _room_target(self, room_name):
        room_poly = self.route_room_polygon(room_name)
        if room_poly is None or room_poly.is_empty:
            return self.terminals.get(room_name)
        centroid = room_poly.centroid
        if room_poly.contains(centroid):
            return float(centroid.x), float(centroid.y)
        return self.representative_point(room_poly)

    def _rotation_score(self, cx, cy, angle):
        shaft_point = self.representative_point(self.shaft) if self.shaft else None
        return score_rotation_field_at(
            self.machine_pins(cx, cy, angle),
            angle,
            self.wet_room_names,
            self.terminals.keys(),
            shaft_point,
            self._room_target,
            self.weight_mode,
            self.local_axis_to_world,
        )
