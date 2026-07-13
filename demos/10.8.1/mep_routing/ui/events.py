"""Pure keyboard-command transitions for the interactive routing UI.

The Pygame loop maps physical keys to these semantic commands and performs
the resulting routing, graph, and placement side effects.  Keeping that
translation here makes the transition policy testable without Pygame.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping


Color = tuple[int, int, int]
Marker = tuple[str, Color]
Point = tuple[float, float]
ScreenPoint = tuple[int, int]


@dataclass(frozen=True)
class RoutingKeyTransition:
    """Updated state and adapter work requested by a routing keyboard command."""

    state: dict[str, int]
    markers: tuple[Marker, ...] = ()
    solve: bool = True
    record_history: bool = True
    rebuild_graph: bool = False
    apply_rotation_mode: bool = False
    refresh_placement_fields: bool = False
    needs_auto_placement: bool = False


def _cycle(value: int, count: int) -> int:
    if count <= 0:
        raise ValueError("option count must be positive")
    return (value + 1) % count


def routing_key_transition(
    command: str,
    state: Mapping[str, int],
    option_counts: Mapping[str, int],
) -> RoutingKeyTransition | None:
    """Return the semantic transition for one solver-control keyboard command.

    ``command`` is intentionally independent from Pygame key constants.  The
    returned flags let the app retain ownership of graph builds, routing, and
    placement calculations while this module owns state-transition policy.
    """
    next_state = dict(state)

    if command == "rotate_machine":
        next_state["machine_angle"] = (next_state["machine_angle"] + 90) % 360
        next_state["auto_placement_mode_idx"] = 0
        return RoutingKeyTransition(
            next_state,
            markers=((f"Rot:{next_state['machine_angle']}", (46, 204, 113)),),
        )
    if command == "cycle_strategy":
        next_state["routing_strategy_idx"] = _cycle(
            next_state["routing_strategy_idx"], option_counts["routing_strategy"]
        )
        return RoutingKeyTransition(next_state)
    if command == "cycle_room_start":
        next_state["room_start_mode_idx"] = _cycle(
            next_state["room_start_mode_idx"], option_counts["room_start"]
        )
        return RoutingKeyTransition(
            next_state,
            markers=(
                (f"Start:{next_state['room_start_mode_idx']}", (155, 89, 182)),
                (f"Strat:{next_state['routing_strategy_idx']}", (52, 152, 219)),
            ),
        )
    if command == "cycle_router_backend":
        next_state["router_backend_idx"] = _cycle(
            next_state["router_backend_idx"], option_counts["router_backend"]
        )
        return RoutingKeyTransition(
            next_state,
            markers=((f"R:{next_state['router_backend_idx']}", (230, 126, 34)),),
        )
    if command == "cycle_heuristic":
        next_state["heuristic_mode_idx"] = _cycle(
            next_state["heuristic_mode_idx"], option_counts["heuristic"]
        )
        return RoutingKeyTransition(
            next_state,
            markers=((f"Heur:{next_state['heuristic_mode_idx']}", (241, 196, 15)),),
        )
    if command == "cycle_graph":
        next_state["graph_type_idx"] = _cycle(next_state["graph_type_idx"], option_counts["graph"])
        return RoutingKeyTransition(
            next_state,
            markers=((f"Grid:{next_state['graph_type_idx']}", (155, 89, 182)),),
            rebuild_graph=True,
        )
    if command == "cycle_rotation_mode":
        next_state["rotation_mode_idx"] = _cycle(
            next_state["rotation_mode_idx"], option_counts["rotation_mode"]
        )
        return RoutingKeyTransition(
            next_state,
            markers=((f"RotMode:{next_state['rotation_mode_idx']}", (95, 178, 218)),),
            apply_rotation_mode=True,
        )
    if command == "toggle_auto_placement":
        next_state["auto_placement_mode_idx"] = 0 if next_state["auto_placement_mode_idx"] > 0 else 2
        return RoutingKeyTransition(
            next_state,
            record_history=False,
            needs_auto_placement=next_state["auto_placement_mode_idx"] > 0,
        )
    if command == "cycle_auto_placement":
        next_state["auto_placement_mode_idx"] = _cycle(
            next_state["auto_placement_mode_idx"], option_counts["auto_placement"]
        )
        return RoutingKeyTransition(
            next_state,
            record_history=False,
            needs_auto_placement=next_state["auto_placement_mode_idx"] > 0,
        )
    if command == "cycle_weight_mode":
        next_state["weight_mode_idx"] = _cycle(next_state["weight_mode_idx"], option_counts["weight_mode"])
        return RoutingKeyTransition(
            next_state,
            markers=((f"W:{'Eq' if next_state['weight_mode_idx'] == 1 else 'Def'}", (241, 196, 15)),),
            refresh_placement_fields=next_state["auto_placement_mode_idx"] == 0,
            needs_auto_placement=next_state["auto_placement_mode_idx"] > 0,
        )
    return None


@dataclass(frozen=True)
class CanvasGestureState:
    """Transient canvas gesture state, independent from Pygame events."""

    ruler_mode: bool = False
    terminal_tool_mode: str | None = None
    panning: bool = False
    pan_last_screen: ScreenPoint | None = None
    ruler_dragging: bool = False
    ruler_start_mm: Point | None = None
    ruler_end_mm: Point | None = None
    terminal_area_dragging: bool = False
    terminal_area_start_mm: Point | None = None
    terminal_area_end_mm: Point | None = None
    terminal_area_remove: bool = False
    machine_dragging: bool = False
    machine_drag_offset_mm: Point | None = None


@dataclass(frozen=True)
class CanvasHit:
    """World-space hit-test results supplied by the application adapter."""

    machine_hit: bool = False
    route_name: str | None = None
    room_route_name: str | None = None


@dataclass(frozen=True)
class CanvasGestureCommand:
    """A semantic canvas action for the application to execute."""

    name: str
    value: object | None = None


@dataclass(frozen=True)
class CanvasGestureTransition:
    state: CanvasGestureState
    commands: tuple[CanvasGestureCommand, ...] = ()


def begin_canvas_gesture(
    state: CanvasGestureState,
    *,
    world_point: Point,
    screen_point: ScreenPoint,
    shift: bool,
    ctrl: bool,
    hit: CanvasHit,
    machine_center_mm: Point | None = None,
) -> CanvasGestureTransition:
    """Start one canvas gesture using precomputed app hit-test results."""
    if shift:
        return CanvasGestureTransition(
            replace(state, panning=True, pan_last_screen=screen_point),
            (CanvasGestureCommand("start_pan"),),
        )
    if state.ruler_mode:
        return CanvasGestureTransition(
            replace(state, ruler_dragging=True, ruler_start_mm=world_point, ruler_end_mm=world_point),
            (CanvasGestureCommand("start_ruler", world_point),),
        )
    if state.terminal_tool_mode == "point":
        return CanvasGestureTransition(
            state,
            (CanvasGestureCommand("apply_terminal_point", (world_point, ctrl)),),
        )
    if state.terminal_tool_mode == "area":
        return CanvasGestureTransition(
            replace(
                state,
                terminal_area_dragging=True,
                terminal_area_start_mm=world_point,
                terminal_area_end_mm=world_point,
                terminal_area_remove=ctrl,
            ),
            (CanvasGestureCommand("start_terminal_area", (world_point, ctrl)),),
        )
    if hit.machine_hit:
        if machine_center_mm is None:
            raise ValueError("machine center is required for a machine-drag hit")
        offset = (world_point[0] - machine_center_mm[0], world_point[1] - machine_center_mm[1])
        return CanvasGestureTransition(
            replace(state, machine_dragging=True, machine_drag_offset_mm=offset),
            (CanvasGestureCommand("start_machine_drag", offset),),
        )
    if hit.route_name is not None:
        return CanvasGestureTransition(state, (CanvasGestureCommand("select_route", hit.route_name),))
    if hit.room_route_name is not None:
        return CanvasGestureTransition(state, (CanvasGestureCommand("select_route", hit.room_route_name),))
    return CanvasGestureTransition(state, (CanvasGestureCommand("clear_route_selection"),))


def move_canvas_gesture(
    state: CanvasGestureState,
    *,
    world_point: Point,
    screen_point: ScreenPoint,
) -> CanvasGestureTransition:
    """Advance the active canvas gesture and emit its adapter action."""
    if state.panning and state.pan_last_screen is not None:
        dx = screen_point[0] - state.pan_last_screen[0]
        dy = screen_point[1] - state.pan_last_screen[1]
        return CanvasGestureTransition(
            replace(state, pan_last_screen=screen_point),
            (CanvasGestureCommand("pan_by", (dx, dy)),),
        )
    if state.ruler_dragging:
        return CanvasGestureTransition(
            replace(state, ruler_end_mm=world_point),
            (CanvasGestureCommand("update_ruler", world_point),),
        )
    if state.terminal_area_dragging:
        return CanvasGestureTransition(
            replace(state, terminal_area_end_mm=world_point),
            (CanvasGestureCommand("update_terminal_area", world_point),),
        )
    if state.machine_dragging:
        if state.machine_drag_offset_mm is None:
            raise ValueError("machine drag requires a drag offset")
        offset_x, offset_y = state.machine_drag_offset_mm
        machine_center = (world_point[0] - offset_x, world_point[1] - offset_y)
        return CanvasGestureTransition(
            state,
            (CanvasGestureCommand("move_machine", machine_center),),
        )
    return CanvasGestureTransition(state)


def end_canvas_gesture(
    state: CanvasGestureState,
    *,
    button: str,
) -> CanvasGestureTransition:
    """Finish a left or middle canvas gesture and request any terminal-area commit."""
    commands: tuple[CanvasGestureCommand, ...] = ()
    if (
        button == "left"
        and state.terminal_area_dragging
        and state.terminal_area_start_mm is not None
        and state.terminal_area_end_mm is not None
    ):
        commands = (
            CanvasGestureCommand(
                "apply_terminal_area",
                (state.terminal_area_start_mm, state.terminal_area_end_mm, state.terminal_area_remove),
            ),
        )
    return CanvasGestureTransition(
        replace(
            state,
            panning=False,
            pan_last_screen=None,
            ruler_dragging=False,
            terminal_area_dragging=False,
            machine_dragging=False,
            machine_drag_offset_mm=None,
        ),
        commands,
    )
