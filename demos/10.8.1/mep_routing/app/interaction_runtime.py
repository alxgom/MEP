"""Pygame-free side-effect sequencing for interactive routing commands."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence


Color = tuple[int, int, int]


@dataclass(frozen=True)
class RoutingRun:
    routes: Any
    status: str
    elapsed_ms: float
    total_nodes: int

    @property
    def is_successful(self) -> bool:
        return bool(self.routes) and not self.status.startswith("Blocked")


@dataclass(frozen=True)
class InteractionState:
    run: RoutingRun
    selected_route_name: str | None = None
    needs_auto_placement: bool = False


@dataclass(frozen=True)
class RoutingTransition:
    needs_auto_placement: bool = False
    record_history: bool = False
    markers: Sequence[tuple[str, Color]] = ()


@dataclass(frozen=True)
class InteractionCallbacks:
    """App-owned actions required to execute a normalized interaction command."""

    solve: Callable[[], RoutingRun]
    record_solution: Callable[[RoutingRun, str, Color], None]
    record_history: Callable[[RoutingRun], None]
    add_marker: Callable[[str, Color], None]
    apply_slider: Callable[[str, float], tuple[str, Color]]
    apply_terminal_point: Callable[[tuple[float, float], bool], tuple[bool, str | None]]
    apply_terminal_area: Callable[[tuple[float, float], tuple[float, float], bool], tuple[bool, str | None]]
    clear_terminal_preferences: Callable[[], None]
    rotate_machine: Callable[[int], int]
    generate_dwelling: Callable[[], None]
    clear_solution_logs: Callable[[], None]
    clear_history: Callable[[], None]
    run_auto_placement: Callable[[], None]


def reroute(
    state: InteractionState,
    callbacks: InteractionCallbacks,
    *,
    record_history: bool = False,
    marker: tuple[str, Color] | None = None,
    solution_marker: tuple[str, Color] | None = None,
) -> InteractionState:
    """Run routing once and apply history/solution effects only for usable routes."""
    run = callbacks.solve()
    updated = replace(state, run=run)
    if not run.is_successful:
        return updated
    if record_history:
        callbacks.record_history(run)
    if marker is not None:
        callbacks.add_marker(*marker)
    if solution_marker is not None:
        callbacks.record_solution(run, *solution_marker)
    return updated


def apply_slider_change(
    state: InteractionState,
    slider_name: str,
    slider_x: float,
    callbacks: InteractionCallbacks,
) -> InteractionState:
    label, color = callbacks.apply_slider(slider_name, slider_x)
    return reroute(state, callbacks, solution_marker=(label, color))


def apply_terminal_point(
    state: InteractionState,
    point: tuple[float, float],
    *,
    remove: bool,
    callbacks: InteractionCallbacks,
) -> InteractionState:
    changed, room_name = callbacks.apply_terminal_point(point, remove)
    updated = replace(state, selected_route_name=room_name or state.selected_route_name)
    if not changed:
        return updated
    return reroute(
        updated,
        callbacks,
        solution_marker=("Term-" if remove else "Term+", (26, 188, 156)),
    )


def apply_terminal_area(
    state: InteractionState,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    *,
    remove: bool,
    callbacks: InteractionCallbacks,
) -> InteractionState:
    changed, room_name = callbacks.apply_terminal_area(start_point, end_point, remove)
    updated = replace(state, selected_route_name=room_name or state.selected_route_name)
    if not changed:
        return updated
    return reroute(
        updated,
        callbacks,
        solution_marker=("Area-" if remove else "Area+", (155, 89, 182)),
    )


def reset_terminal_preferences(
    state: InteractionState,
    callbacks: InteractionCallbacks,
) -> InteractionState:
    callbacks.clear_terminal_preferences()
    return reroute(state, callbacks, solution_marker=("Prefs reset", (26, 188, 156)))


def apply_rotation(
    state: InteractionState,
    delta_degrees: int,
    callbacks: InteractionCallbacks,
) -> InteractionState:
    angle = callbacks.rotate_machine(delta_degrees)
    return reroute(
        state,
        callbacks,
        record_history=True,
        marker=(f"Rot:{angle}", (46, 204, 113)),
    )


def apply_routing_transition(
    state: InteractionState,
    transition: RoutingTransition,
    callbacks: InteractionCallbacks,
) -> InteractionState:
    updated = replace(
        state,
        needs_auto_placement=state.needs_auto_placement or transition.needs_auto_placement,
    )
    updated = reroute(updated, callbacks, record_history=transition.record_history)
    if updated.run.is_successful and transition.record_history:
        for marker in transition.markers:
            callbacks.add_marker(*marker)
    return updated


def apply_dwelling_change(
    state: InteractionState,
    callbacks: InteractionCallbacks,
    *,
    marker: tuple[str, Color] | None = None,
    clear_history: bool = True,
    needs_auto_placement: bool = False,
) -> InteractionState:
    callbacks.generate_dwelling()
    callbacks.clear_solution_logs()
    if clear_history:
        callbacks.clear_history()
    updated = replace(state, needs_auto_placement=needs_auto_placement)
    return reroute(updated, callbacks, record_history=True, marker=marker)


def apply_auto_placement(
    state: InteractionState,
    callbacks: InteractionCallbacks,
) -> InteractionState:
    callbacks.run_auto_placement()
    updated = replace(state, needs_auto_placement=False)
    return reroute(updated, callbacks, record_history=True, marker=("Auto", (230, 126, 34)))
