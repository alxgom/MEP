"""Cli-specific PEAD placement policy."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .geometry import (
    cli_clear_supply_turns, cli_machine_footprint, cli_machine_pins, cli_supply_access_specs,
    cli_supply_exit_dimensions, cli_supply_exit_footprints,
)


@dataclass(frozen=True)
class CliPlacementResult:
    center: tuple[int, int]
    angle_degrees: int
    score: float
    feasible_candidates: int
    green_score_gap_mm: float | None = None  # Green minus yellow before the penalty.


class CliPlacementRuntime:
    """Place a selected Cli machine without consuming Sal placement fields."""

    def __init__(self, spacing_mm=400):
        self.spacing_mm = int(spacing_mm)

    def place(self, prepared, machine, grille_plan, *, context_obstacles=(), exit_dimensions=None,
              angle_degrees=None):
        region = prepared.routing_region_base
        context_obstacles = tuple(context_obstacles)
        obstacles = tuple(prepared.columns) + tuple(prepared.shafts) + tuple(context_obstacles)
        length, width = exit_dimensions or cli_supply_exit_dimensions(machine, None, 1.0)
        min_x, min_y, max_x, max_y = region.bounds
        xs = range(math.ceil(min_x / self.spacing_mm) * self.spacing_mm, math.floor(max_x / self.spacing_mm) * self.spacing_mm + 1, self.spacing_mm)
        ys = range(math.ceil(min_y / self.spacing_mm) * self.spacing_mm, math.floor(max_y / self.spacing_mm) * self.spacing_mm + 1, self.spacing_mm)
        candidates = [(x, y) for y in ys for x in xs]
        candidates.append(tuple(map(round, prepared.machine_position)))
        representative = region.representative_point()
        candidates.append((round(float(representative.x)), round(float(representative.y))))
        targets = tuple(grille_plan.connections)
        best = None
        feasible = 0
        angles = (0, 90) if angle_degrees is None else (angle_degrees,)
        for angle in angles:
            for center in dict.fromkeys(candidates):
                pins = cli_machine_pins(machine, center, angle)
                footprint = cli_machine_footprint(pins)
                if not region.covers(footprint) or any(footprint.intersects(obstacle) for obstacle in obstacles):
                    continue
                access, = cli_supply_access_specs(machine, pins, angle, length)
                extension, turns = cli_supply_exit_footprints(machine, angle, access, width)
                if not cli_clear_supply_turns(prepared, extension, turns, context_obstacles):
                    continue
                feasible += 1
                route_distance = sum(
                    math.dist(
                        pins["air_out" if item.role == "supply" else "air_in"],
                        item.routing_point,
                    )
                    for item in targets
                )
                clearance = min((footprint.distance(obstacle) for obstacle in obstacles), default=2000.0)
                score = route_distance - min(clearance, 2000.0) * 0.1
                key = (score, angle, center)
                if best is None or key < best[0]:
                    best = (key, center, angle, score)
        if best is None:
            center = (round(float(representative.x)), round(float(representative.y)))
            return CliPlacementResult(center, angles[0], math.inf, 0)
        return CliPlacementResult(best[1], best[2], best[3], feasible)
