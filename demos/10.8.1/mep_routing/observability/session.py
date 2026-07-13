"""State containers for an interactive routing-observation session.

The session owns only observation state.  Route scoring, snapshots, and UI
rendering remain injected by the application so this module stays independent
from Pygame and routing-domain policy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Iterable, Mapping, MutableMapping, Sequence


HistoryMarker = tuple[int, str, tuple[int, int, int]]
MetricDefinition = tuple[str, str, tuple[int, int, int]]
MetricValue = Callable[[Mapping[str, Any], str], float]


@dataclass
class RoutingHistory:
    """Bounded plot buffers with absolute sample indices for event markers."""

    maxlen: int = 400
    length_m: Deque[float] = field(init=False)
    score: Deque[float] = field(init=False)
    turns: Deque[int] = field(init=False)
    turns_per_m: Deque[float] = field(init=False)
    elapsed_ms: Deque[float] = field(init=False)
    sample_count: int = 0
    event_markers: list[HistoryMarker] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.length_m = deque(maxlen=self.maxlen)
        self.score = deque(maxlen=self.maxlen)
        self.turns = deque(maxlen=self.maxlen)
        self.turns_per_m = deque(maxlen=self.maxlen)
        self.elapsed_ms = deque(maxlen=self.maxlen)

    @property
    def buffers(self) -> tuple[Deque[float], Deque[float], Deque[int], Deque[float], Deque[float]]:
        return self.length_m, self.score, self.turns, self.turns_per_m, self.elapsed_ms

    @property
    def latest_index(self) -> int | None:
        return self.sample_count - 1 if self.length_m else None

    def append(self, sample: Mapping[str, float]) -> int:
        """Store a normalized sample and return its absolute plot index."""
        self.length_m.append(sample["length_m"])
        self.score.append(sample["score"])
        self.turns.append(sample["turns"])
        self.turns_per_m.append(sample["turns_per_m"])
        self.elapsed_ms.append(sample["elapsed_ms"])
        self.sample_count += 1
        return self.sample_count - 1

    def add_marker(self, label: str, color: tuple[int, int, int], index: int | None = None) -> bool:
        marker_index = self.latest_index if index is None else index
        if marker_index is None:
            return False
        self.event_markers.append((marker_index, label, color))
        return True

    def replace_marker(self, label: str, index: int, color: tuple[int, int, int]) -> None:
        self.event_markers[:] = [marker for marker in self.event_markers if marker[1] != label]
        self.event_markers.append((index, label, color))

    def clear(self) -> None:
        for buffer in self.buffers:
            buffer.clear()
        self.event_markers.clear()
        self.sample_count = 0


@dataclass
class SolutionLogSession:
    """Manual and automatic solution-log state plus the selected entry."""

    manual_logs: list[dict[str, Any]] = field(default_factory=list)
    auto_best_logs: MutableMapping[str, dict[str, Any]] = field(default_factory=dict)
    selected_log_id: int | str | None = None

    def add_manual(self, snapshot: Mapping[str, Any], history_index: int | None) -> dict[str, Any]:
        entry = dict(snapshot)
        entry.update(id=len(self.manual_logs) + 1, hist_idx=history_index, kind="manual")
        self.manual_logs.append(entry)
        self.selected_log_id = entry["id"]
        return entry

    def update_auto_bests(
        self,
        snapshot: Mapping[str, Any],
        history_index: int | None,
        metric_definitions: Iterable[MetricDefinition],
        metric_value: MetricValue,
    ) -> list[tuple[str, dict[str, Any], str, tuple[int, int, int]]]:
        """Keep only strictly improved automatic entries for each metric."""
        if history_index is None:
            return []

        updates = []
        for metric, label, color in metric_definitions:
            value = metric_value(snapshot, metric)
            previous = self.auto_best_logs.get(metric)
            if previous is not None and value >= metric_value(previous, metric) - 1e-9:
                continue
            entry = dict(snapshot)
            entry.update(id=f"best:{metric}", kind="auto", metric=metric, hist_idx=history_index)
            self.auto_best_logs[metric] = entry
            updates.append((metric, entry, label, color))
        return updates

    def clear(self) -> None:
        self.manual_logs.clear()
        self.auto_best_logs.clear()
        self.selected_log_id = None

