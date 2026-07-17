"""Ordered execution boundary for independently owned installation runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping

from .installation_context import installation_context_obstacles
from .installation_runtime import InstallationRegistry, InstallationSolveContext


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


class InstallationCoordinator:
    """Execute registered installations in routing order and own result publication."""

    def __init__(self, registry: InstallationRegistry, workbench_state):
        self.registry = registry
        self.workbench_state = workbench_state

    def execute(self, order, enabled, active, dwelling):
        runs = {}
        completed = []
        for key in order:
            runtime = self.registry.get(key)
            if key not in enabled or runtime is None:
                continue
            state = self.workbench_state.installation(key)
            result = runtime.solve(InstallationSolveContext(
                dwelling, state, installation_context_obstacles(completed),
            ))
            runs[key] = result
            completed.append(SimpleNamespace(
                routes=getattr(result, "routes", ()),
                machine_pins=dict(runtime.machine_pins(state)),
            ))
        self.workbench_state.replace_runs(runs)
        if active not in enabled:
            return InstallationExecution(runs, None, f"Routing disabled: {active.title()}")
        result = runs.get(active)
        status = getattr(result, "status", f"Routing unavailable: {active.title()}")
        return InstallationExecution(runs, result, status)
