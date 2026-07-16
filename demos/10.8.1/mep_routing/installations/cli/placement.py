"""Cli-specific PEAD placement policy."""

from __future__ import annotations

from dataclasses import dataclass
import math

from shapely.geometry import Point, Polygon

from .geometry import cli_machine_pins


@dataclass(frozen=True)
class CliPlacementResult:
    center: tuple[int, int]
    angle_degrees: int
    score: float
    feasible_candidates: int


class CliPlacementRuntime:
    """Place a selected Cli machine without consuming Sal placement fields."""

    def __init__(self, spacing_mm=400):
        self.spacing_mm = int(spacing_mm)

    def place(self, prepared, machine, grille_plan, *, context_obstacles=()):
        region = prepared.routing_region_base
        obstacles = tuple(prepared.columns) + tuple(prepared.shafts) + tuple(context_obstacles)
        min_x, min_y, max_x, max_y = region.bounds
        xs = range(math.ceil(min_x / self.spacing_mm) * self.spacing_mm, math.floor(max_x / self.spacing_mm) * self.spacing_mm + 1, self.spacing_mm)
        ys = range(math.ceil(min_y / self.spacing_mm) * self.spacing_mm, math.floor(max_y / self.spacing_mm) * self.spacing_mm + 1, self.spacing_mm)
        candidates = [(x, y) for y in ys for x in xs]
        candidates.append(tuple(map(round, prepared.machine_position)))
        representative = region.representative_point()
        candidates.append((round(float(representative.x)), round(float(representative.y))))
        targets = [item.routing_point for item in grille_plan.connections]
        best = None
        feasible = 0
        for angle in (0, 90):
            for center in dict.fromkeys(candidates):
                pins = cli_machine_pins(machine, center, angle)
                footprint = Polygon([pins[name] for name in ("c_tl", "c_tr", "c_br", "c_bl")])
                if not region.covers(footprint) or any(footprint.intersects(obstacle) for obstacle in obstacles):
                    continue
                feasible += 1
                route_distance = sum(math.dist(center, target) for target in targets)
                clearance = min((footprint.distance(obstacle) for obstacle in obstacles), default=2000.0)
                score = route_distance - min(clearance, 2000.0) * 0.1
                key = (score, angle, center)
                if best is None or key < best[0]:
                    best = (key, center, angle, score)
        if best is None:
            center = (round(float(representative.x)), round(float(representative.y)))
            return CliPlacementResult(center, 0, math.inf, 0)
        return CliPlacementResult(best[1], best[2], best[3], feasible)
