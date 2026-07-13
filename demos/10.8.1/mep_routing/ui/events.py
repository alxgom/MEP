"""Pure keyboard-command transitions for the interactive routing UI.

The Pygame loop maps physical keys to these semantic commands and performs
the resulting routing, graph, and placement side effects.  Keeping that
translation here makes the transition policy testable without Pygame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


Color = tuple[int, int, int]
Marker = tuple[str, Color]


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
