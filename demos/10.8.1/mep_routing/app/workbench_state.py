"""Mutable application state for the multi-installation workbench.

The objects here contain values only.  Solver, Pygame, and filesystem services
remain owned by the application composition layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MachinePose:
    x_mm: float = 0.0
    y_mm: float = 0.0
    angle_degrees: float = 0.0

    @property
    def center(self) -> tuple[float, float]:
        return self.x_mm, self.y_mm

    def set(self, center, angle_degrees=None) -> None:
        self.x_mm, self.y_mm = center
        if angle_degrees is not None:
            self.angle_degrees = angle_degrees % 360

    def rotate(self, delta_degrees: float) -> float:
        self.angle_degrees = (self.angle_degrees + delta_degrees) % 360
        return self.angle_degrees

    def as_tuple(self) -> tuple[float, float, float]:
        return self.x_mm, self.y_mm, self.angle_degrees


@dataclass
class InstallationState:
    key: str
    pose: MachinePose = field(default_factory=MachinePose)
    result: Any = None


@dataclass
class DwellingState:
    active: Any = None
    workspace: Any = None
    graph_build_session: Any = None
    terminal_runtime: Any = None
    machine_vertical_clearance_blocks: tuple = ()
    scenario_label: str = "synthetic"
    scenario_summary: dict = field(default_factory=dict)


@dataclass
class SalWorkbenchState:
    """Installation-specific interactive choices, migrated incrementally."""

    graph_type_idx: int = 0
    routing_strategy_idx: int = 0
    router_backend_idx: int = 0
    heuristic_mode_idx: int = 0
    auto_placement_mode_idx: int = 2
    rotation_mode_idx: int = 0
    room_start_mode_idx: int = 0
    weight_mode_idx: int = 0


@dataclass
class CliWorkbenchState:
    """Cli-specific interactive choices not yet available in routing core."""

    return_ventilation_enabled: bool = False
    graph_type_idx: int = 3
    routing_strategy_idx: int = 0
    router_backend_idx: int = 0
    heuristic_mode_idx: int = 0
    room_start_mode_idx: int = 0
    auto_placement_mode_idx: int = 1
    bend_penalty_mm: float = 4000.0
    crossing_penalty_multiplier: float = 5.0
    minimum_segment_length_factor: float = 1.0
    duct_geometry_mode_idx: int = 0


@dataclass
class CocWorkbenchState:
    """Coc owns a fixed two-terminal routing choice, without placement state."""

    graph_type_idx: int = 0
    routing_strategy_idx: int = 0
    router_backend_idx: int = 0
    heuristic_mode_idx: int = 0
    room_start_mode_idx: int = 0


@dataclass
class DisplayState:
    show_grid_graph: bool = False
    show_heatmap: bool = False
    edge_weight_heatmap_enabled: bool = False
    edge_weight_view_mode_idx: int = 0
    route_real_diameter_width_enabled: bool = False


@dataclass
class WorkbenchState:
    installations: dict[str, InstallationState]
    dwelling: DwellingState = field(default_factory=DwellingState)
    sal: SalWorkbenchState = field(default_factory=SalWorkbenchState)
    cli: CliWorkbenchState = field(default_factory=CliWorkbenchState)
    coc: CocWorkbenchState = field(default_factory=CocWorkbenchState)
    display: DisplayState = field(default_factory=DisplayState)
    solver_dropdown_open_for: tuple[str, str] | None = None

    @classmethod
    def create(cls, installation_keys=("sal", "cli"), **kwargs):
        return cls(
            installations={key: InstallationState(key) for key in installation_keys},
            **kwargs,
        )

    def installation(self, key: str) -> InstallationState:
        return self.installations[key]

    @property
    def runs(self) -> dict[str, Any]:
        return {
            key: installation.result
            for key, installation in self.installations.items()
            if installation.result is not None
        }

    def replace_runs(self, runs) -> None:
        for installation in self.installations.values():
            installation.result = None
        for key, result in runs.items():
            if key in self.installations:
                self.installations[key].result = result
