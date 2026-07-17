"""Immutable visual theme for the Demo workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

Color = tuple[int, int, int] | tuple[int, int, int, int]


def _route_colors() -> Mapping[str, Color]:
    return MappingProxyType({
        "Shaft": (192, 57, 52), "Kitchen": (241, 196, 15),
        "Bathroom": (52, 152, 219), "Bathroom 1": (52, 152, 219),
        "Bathroom 2": (155, 89, 182), "Toilet": (230, 126, 34),
        "Washroom": (26, 188, 156), "Supply Air Tree": (52, 152, 219),
        "Return Air Tree": (155, 89, 182),
    })


@dataclass(frozen=True)
class WorkbenchTheme:
    installation_header_top: int = 4
    panel_width: int = 280
    colorbar_width: int = 56
    wall_draw_width: int = 2
    background: Color = (240, 238, 233)
    panel: Color = (29, 30, 39)
    plot_background: Color = (22, 22, 30)
    room: Color = (45, 52, 54)
    covered_room: Color = (248, 247, 243)
    cover_overlay: Color = (96, 104, 112, 34)
    hallway: Color = (57, 72, 85)
    wall: Color = (0, 0, 0)
    wet_room_accent: Color = (95, 178, 218)
    column: Color = (0, 0, 0)
    shaft: Color = (231, 76, 60)
    shaft_background: Color = (231, 76, 60, 40)
    inactive_shaft: Color = (154, 84, 82)
    inactive_shaft_hatch: Color = (92, 58, 58)
    machine_clearance_hatch: Color = (105, 105, 105)
    machine_clearance_fill: Color = (145, 145, 145, 235)
    door: Color = (220, 220, 220)
    machine: Color = (127, 140, 141)
    machine_hover: Color = (149, 165, 166)
    text: Color = (236, 240, 241)
    muted: Color = (149, 165, 166)
    deselected_route: Color = (7, 8, 10)
    deselected_room: Color = (39, 42, 43)
    deselected_pin: Color = (112, 116, 120)
    selection_halo: Color = (210, 235, 255)
    blocked_edge: Color = (82, 76, 88)
    graph_edge: Color = (122, 130, 140)
    graph_node: Color = (96, 106, 116)
    plan_label: Color = (18, 22, 28)
    plan_label_halo: Color = (255, 255, 255)
    terminal_allowed: Color = (78, 158, 178)
    terminal_blocked: Color = (142, 78, 86)
    terminal_blocked_hatch: Color = (96, 54, 62)
    route_colors: Mapping[str, Color] = field(default_factory=_route_colors)

    def chrome_colors(self) -> dict[str, Color]:
        return {
            "plot_background": self.plot_background, "text": self.text,
            "muted": self.muted, "plan_label": self.plan_label,
            "wet_room": self.wet_room_accent, "wall": self.wall,
        }
