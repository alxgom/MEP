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
    routing_weights: bool = False


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
    color: tuple[int, int, int] = (105, 113, 124)

    @property
    def routes(self):
        return getattr(self.result, "routes", ()) or ()

    @property
    def terminal_segments(self):
        plan = getattr(self.result, "grille_plan", None)
        connections = getattr(plan, "connections", ()) if plan is not None else ()
        result = {}
        for item in connections:
            center_x, center_y = item.grille_point
            normal_x, normal_y = item.inward_normal
            tangent_x, tangent_y = -float(normal_y), float(normal_x)
            magnitude = (tangent_x * tangent_x + tangent_y * tangent_y) ** 0.5
            if magnitude <= 1e-9:
                tangent_x, tangent_y, magnitude = 1.0, 0.0, 1.0
            half = float(item.width_mm) / 2.0
            dx, dy = tangent_x / magnitude * half, tangent_y / magnitude * half
            result[item.route_name] = tuple(sorted((
                (center_x - dx, center_y - dy),
                (center_x + dx, center_y + dy),
            )))
        return result

    def can(self, name):
        return self.active and bool(getattr(self.capabilities, name))


@dataclass(frozen=True)
class WorkbenchProjection:
    active: InstallationProjection
    visible: tuple[InstallationProjection, ...]

    @property
    def context_layers(self):
        return tuple(
            CanvasContextLayer(
                item.key, item.terminals, item.routes, item.pins, item.color,
                item.terminal_segments,
            )
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
            getattr(app, "theme", SimpleNamespace(installation_colors={})).installation_colors.get(
                key, (105, 113, 124),
            ),
        ))
    active = next((item for item in items if item.active), None)
    if active is None:
        runtime = app.routing_controller.installations.get(active_key)
        state = app.workbench_state.installations[active_key]
        active = InstallationProjection(
            active_key, True, ProjectedCapabilities(),
            SimpleNamespace(routes=(), status=f"Routing disabled: {active_key.title()}", elapsed_ms=0.0, total_nodes=0),
            dwelling.workspace, app.terminals, state.pose.as_tuple(),
            {}, getattr(app, "theme", SimpleNamespace(installation_colors={})).installation_colors.get(
                active_key, (105, 113, 124),
            ),
        )
    return WorkbenchProjection(active, tuple(items))


def installation_context_layers(order, enabled, active, runs, visuals):
    return tuple(
        CanvasContextLayer(key, visuals[key].terminals, runs[key].routes or (), visuals[key].machine_pins)
        for key in order
        if key != active and key in enabled and key in runs and key in visuals
    )
