"""Installation-neutral projection of non-active runs onto the canvas."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping

from .frame import CanvasContextLayer


@dataclass(frozen=True)
class InstallationVisual:
    terminals: Mapping[str, tuple[float, float]]
    machine_pins: Mapping[str, tuple[float, float]]


@dataclass(frozen=True)
class ProjectedCapabilities:
    machine_selection: bool = False
    automatic_placement: bool = False
    terminal_tools: bool = False
    machine_edit: bool = False
    solver_controls: bool = False
    route_validation: bool = False
    placement_overlay: bool = False


@dataclass(frozen=True)
class InstallationProjection:
    key: str
    active: bool
    capabilities: ProjectedCapabilities
    result: object
    workspace: object
    terminals: Mapping
    pose: tuple
    pins: Mapping

    @property
    def routes(self):
        return getattr(self.result, "routes", ()) or ()

    def can(self, name):
        return self.active and bool(getattr(self.capabilities, name))


@dataclass(frozen=True)
class WorkbenchProjection:
    active: InstallationProjection
    visible: tuple[InstallationProjection, ...]

    @property
    def context_layers(self):
        return tuple(
            CanvasContextLayer(item.key, item.terminals, item.routes, item.pins)
            for item in self.visible if not item.active
        )


def active_capabilities(app):
    key = app.installation_header_state.active
    runtime = app.routing_controller.installations.get(key)
    source = getattr(runtime, "capabilities", None)
    enabled = key in app.installation_header_state.enabled and source is not None
    return ProjectedCapabilities(**{
        name: enabled and bool(getattr(source, name, False))
        for name in ProjectedCapabilities.__dataclass_fields__
    })


def project_workbench(app):
    active_key = app.installation_header_state.active
    dwelling = app.workbench_state.dwelling
    items = []
    for key in app.installation_header_state.order:
        if key not in app.installation_header_state.enabled:
            continue
        runtime = app.routing_controller.installations.get(key)
        state = app.workbench_state.installations.get(key)
        if runtime is None or state is None or state.result is None:
            continue
        result = state.result
        workspace = getattr(result, "workspace", None) or dwelling.workspace
        terminals = getattr(result, "terminals", None)
        if terminals is None:
            terminals = app.terminals
        caps = runtime.capabilities
        projected = ProjectedCapabilities(**{
            name: bool(getattr(caps, name, False))
            for name in ProjectedCapabilities.__dataclass_fields__
        })
        items.append(InstallationProjection(
            key, key == active_key, projected, result, workspace, terminals,
            state.pose.as_tuple(), runtime.machine_pins(state),
        ))
    active = next((item for item in items if item.active), None)
    if active is None:
        runtime = app.routing_controller.installations.get(active_key)
        state = app.workbench_state.installations[active_key]
        active = InstallationProjection(
            active_key, True, ProjectedCapabilities(),
            SimpleNamespace(routes=(), status=f"Routing disabled: {active_key.title()}", elapsed_ms=0.0, total_nodes=0),
            dwelling.workspace, app.terminals, state.pose.as_tuple(),
            {} if runtime is None else runtime.machine_pins(state),
        )
    return WorkbenchProjection(active, tuple(items))


def installation_context_layers(order, enabled, active, runs, visuals):
    return tuple(
        CanvasContextLayer(key, visuals[key].terminals, runs[key].routes or (), visuals[key].machine_pins)
        for key in order
        if key != active and key in enabled and key in runs and key in visuals
    )
