"""Prepared Sal routing data shared by each solver strategy branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .policy import SalSolverPolicy
from .route_plan import SalRoutePlan


@dataclass(frozen=True)
class SalPreparedRoutingProblem:
    """Common Sal topology and shaft result after controller preparation."""

    route_plan: SalRoutePlan
    policy: SalSolverPolicy
    global_pins: Mapping[str, Any]
    pin_node_map: Mapping[str, Any]
    shaft_boundary_nodes: tuple[int, ...]
    shaft_node_idx: int
    shaft_path: Any
    chosen_shaft_pin: str
    chosen_shaft_target: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "shaft_boundary_nodes", tuple(self.shaft_boundary_nodes))
