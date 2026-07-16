"""Ordered execution boundary for independently owned installation runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Mapping

from .installation_context import installation_context_obstacles


@dataclass(frozen=True)
class InstallationRunSpec:
    solve: Callable[[tuple], object]
    machine_pins: Callable[[], Mapping[str, tuple[int, int]]]


@dataclass(frozen=True)
class InstallationExecution:
    runs: Mapping[str, object]
    active_result: object | None
    active_status: str

    def routing_tuple(self):
        if self.active_result is None:
            return [], self.active_status, 0.0, 0
        result = self.active_result
        return result.routes, result.status, result.elapsed_ms, result.total_nodes


def execute_installations(order, enabled, active, specs):
    """Run enabled installations in order, projecting earlier geometry as context."""
    runs = {}
    completed = []
    for key in order:
        if key not in enabled or key not in specs:
            continue
        spec = specs[key]
        result = spec.solve(installation_context_obstacles(completed))
        runs[key] = result
        completed.append(SimpleNamespace(
            routes=getattr(result, "routes", ()), machine_pins=dict(spec.machine_pins()),
        ))
    if active not in enabled:
        return InstallationExecution(runs, None, f"Routing disabled: {active.title()}")
    result = runs.get(active)
    status = getattr(result, "status", f"Routing unavailable: {active.title()}")
    return InstallationExecution(runs, result, status)
