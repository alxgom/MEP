"""Concrete network and machine-port plan for the current Sal installation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .routes import KITCHEN_ROUTE_NAME, SHAFT_ROUTE_NAME


SAL_SMALL_ROUTE_KEYWORDS = ("Bathroom", "Toilet", "Washroom")
SAL_LARGE_PORTS = ("left_mid", "right_mid")
SAL_SMALL_PORTS = ("tl", "tr", "bl", "br")


@dataclass(frozen=True)
class SalRoutePlan:
    """Sal's existing shaft, kitchen, and wet-room routing topology."""

    small_routes: tuple[str, ...]
    has_kitchen: bool
    shaft_route: str = SHAFT_ROUTE_NAME
    kitchen_route: str = KITCHEN_ROUTE_NAME
    large_ports: tuple[str, str] = SAL_LARGE_PORTS
    small_ports: tuple[str, ...] = SAL_SMALL_PORTS

    @property
    def large_routes(self) -> tuple[str, str]:
        return self.shaft_route, self.kitchen_route

    @property
    def all_routes(self) -> tuple[str, ...]:
        return self.large_routes + self.small_routes

    def kitchen_port_for(self, shaft_port: str) -> str:
        if shaft_port not in self.large_ports:
            raise ValueError(f"Unknown Sal large port: {shaft_port}")
        return self.large_ports[1] if shaft_port == self.large_ports[0] else self.large_ports[0]


def build_sal_route_plan(
    terminals: Mapping[str, object],
    machine_center: tuple[float, float],
) -> SalRoutePlan:
    """Build the route order currently shared by every Sal solver strategy."""
    cx, cy = machine_center
    small_routes = tuple(sorted(
        (
            name
            for name in terminals
            if name != KITCHEN_ROUTE_NAME
            and any(keyword in name for keyword in SAL_SMALL_ROUTE_KEYWORDS)
        ),
        key=lambda name: math.hypot(
            float(terminals[name][0]) - float(cx),
            float(terminals[name][1]) - float(cy),
        ),
    ))
    return SalRoutePlan(
        small_routes=small_routes,
        has_kitchen=KITCHEN_ROUTE_NAME in terminals,
    )
