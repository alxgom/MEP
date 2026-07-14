"""Concrete solver policy values used by the current Sal installation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SalSolverPolicy:
    """Immutable snapshot of Sal's live solver and edge-weight settings.

    UI ranges and graph/terminal feasibility settings intentionally remain
    outside this policy.  The derived score penalties mirror the relationships
    currently maintained by the interactive application.
    """

    bend_cost: float
    crossing_penalty_multiplier: float
    duct_buffer_ratio: float
    shaft_clearance_mm: float
    machine_clearance_soft_margin_mm: float
    overlap_block_weight: float
    min_piece_factor: float
    heuristic_mode: int
    negotiated_iterations: int = 20
    negotiated_present_penalty: float = 20_000.0
    negotiated_history_penalty: float = 4_000.0
    negotiated_large_route_factor: float = 0.35

    @property
    def crossing_penalty(self) -> float:
        return self.crossing_penalty_multiplier * self.bend_cost

    @property
    def clearance_penalty(self) -> float:
        return self.crossing_penalty

    @property
    def overlap_score_penalty(self) -> float:
        return 50.0 * self.bend_cost

    @property
    def short_piece_score_penalty(self) -> float:
        return 2.0 * self.bend_cost
