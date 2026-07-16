"""Application-facing history and solution-log orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .history import history_sample
from .session import RoutingHistory, SolutionLogSession
from .snapshots import solution_snapshot


@dataclass
class RoutingObservationRuntime:
    """Own one interactive run's plots and reproducible solution logs."""

    route_length_m: Callable[[Mapping[str, Any]], float]
    route_turns: Callable[[Mapping[str, Any]], int]
    route_score: Callable[[Mapping[str, Any], int], float]
    solution_kpis: Callable[[Mapping[str, Any], float], Mapping[str, Any]]
    metric_value: Callable[[Mapping[str, Any], str], float]
    best_metrics: Sequence[tuple[str, str, tuple[int, int, int]]]
    history: RoutingHistory = field(default_factory=RoutingHistory)
    logs: SolutionLogSession = field(default_factory=SolutionLogSession)

    def record(self, routes, crossings_count: int, elapsed_ms: float) -> None:
        self.history.append(history_sample(
            routes, crossings_count, elapsed_ms,
            self.route_length_m, self.route_turns, self.route_score,
        ))

    def snapshot(self, state: Mapping[str, Any], routes, status: str, elapsed_ms: float, total_nodes: int):
        return solution_snapshot(
            state, self.solution_kpis(routes, elapsed_ms), status, total_nodes,
        )

    def log_manual(self, state, routes, status: str, elapsed_ms: float, total_nodes: int) -> bool:
        if not routes:
            return False
        self.record(routes, self.solution_kpis(routes, elapsed_ms)["crossings"], elapsed_ms)
        self.update_auto_bests(state, routes, "Auto best", elapsed_ms, 0)
        entry = self.logs.add_manual(
            self.snapshot(state, routes, status, elapsed_ms, total_nodes),
            self.history.latest_index,
        )
        self.history.add_marker(f"L{entry['id']}", (255, 255, 255))
        return True

    def update_auto_bests(self, state, routes, status: str, elapsed_ms: float, total_nodes: int) -> None:
        if not routes:
            return
        snapshot = self.snapshot(state, routes, status, elapsed_ms, total_nodes)
        for _metric, _entry, label, color in self.logs.update_auto_bests(
            snapshot, self.history.latest_index, self.best_metrics, self.metric_value,
        ):
            self.history.replace_marker(label, self.history.latest_index, color)

    def clear(self) -> None:
        self.history.clear()
