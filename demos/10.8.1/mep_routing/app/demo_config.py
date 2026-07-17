"""Typed, immutable composition defaults for the Demo 10.8.1 workbench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mep_routing.config import SALUBRIDAD_DEFAULTS

from .dwelling_application import CommonGraphConfig, TerminalRuntimeConfig


@dataclass(frozen=True)
class DemoConfig:
    scale_to_mm: int
    wall_thickness_mm: int
    grid_spacing_mm: int
    hannan_scaffold_spacing_mm: int
    core_epsilon_grid_mm: int
    machine_clearance_mm: int
    duct_buffer_ratio: float
    routing_wall_clearance_mm: int
    terminal_clearance_mm: int
    terminal_room_buffer_mm: int
    shaft_clearance_mm: int
    shaft_entry_search_mm: int
    shaft_entry_max_candidates: int
    machine_clearance_soft_margin_mm: int
    fps: int
    wheel_rotate_cooldown_ms: int
    bend_default: float
    bend_min: float
    bend_max: float
    crossing_default: float
    crossing_min: float
    crossing_max: float
    overlap_block_weight: float
    min_piece_factor_default: float
    min_piece_factor_min: float
    min_piece_factor_max: float
    graph_types: tuple[str, ...]
    routing_strategies: tuple[str, ...]
    router_backends: tuple[str, ...]
    heuristic_modes: tuple[str, ...]
    auto_placement_modes: tuple[str, ...]
    rotation_modes: tuple[str, ...]
    room_start_modes: tuple[str, ...]
    dwelling_source_modes: tuple[str, ...]
    routing_frame_options: tuple[str, ...]
    rotation_field_epsilon: float
    preferred_terminal_marker_size_mm: float
    preferred_terminal_remap_tolerance_mm: float
    history_max_length: int
    dwelling_export_root: Path
    real_dwelling_db: Path
    preferred_shaft_installation: str

    @property
    def common_graph(self) -> CommonGraphConfig:
        return CommonGraphConfig(
            self.grid_spacing_mm,
            self.hannan_scaffold_spacing_mm,
            self.wall_thickness_mm,
            self.routing_wall_clearance_mm,
            self.core_epsilon_grid_mm,
        )

    @property
    def terminal_runtime(self) -> TerminalRuntimeConfig:
        return TerminalRuntimeConfig(
            self.terminal_clearance_mm,
            self.terminal_room_buffer_mm,
            self.preferred_terminal_remap_tolerance_mm,
        )


def build_demo_config(sal_installation, catalog=SALUBRIDAD_DEFAULTS) -> DemoConfig:
    """Build checked-in workbench defaults around one concrete Sal definition."""
    default = catalog.get_default
    return DemoConfig(
        scale_to_mm=1000,
        wall_thickness_mm=default("WALL_THICKNESS"),
        grid_spacing_mm=default("GRID_SPACING"),
        hannan_scaffold_spacing_mm=default("HANNAN_SCAFFOLD_SPACING"),
        core_epsilon_grid_mm=default("CORE_EPSILON_GRID_MM"),
        machine_clearance_mm=0,
        duct_buffer_ratio=default("DUCT_BUFFER_RATIO"),
        routing_wall_clearance_mm=default("ROUTING_WALL_CLEARANCE_MM"),
        terminal_clearance_mm=default("TERMINAL_REGULATION_CLEARANCE_MM"),
        terminal_room_buffer_mm=default("BUFFER_ROOM_TERMINALES_AIRE_MM"),
        shaft_clearance_mm=default("PATINEJO_CLEARANCE_MM"),
        shaft_entry_search_mm=default("SHAFT_ENTRY_SEARCH_MM"),
        shaft_entry_max_candidates=default("SHAFT_ENTRY_MAX_CANDIDATES"),
        machine_clearance_soft_margin_mm=default("MACHINE_CLEARANCE_SOFT_MARGIN_MM"),
        fps=default("FPS"), wheel_rotate_cooldown_ms=180,
        bend_default=default("C_BEND_DEFAULT"),
        bend_min=default("C_BEND_MIN"), bend_max=default("C_BEND_MAX"),
        crossing_default=default("CROSSING_MULTIPLIER_DEFAULT"),
        crossing_min=default("CROSSING_MULTIPLIER_MIN"),
        crossing_max=default("CROSSING_MULTIPLIER_MAX"),
        overlap_block_weight=default("OVERLAP_BLOCK_WEIGHT"),
        min_piece_factor_default=default("MIN_PIECE_FACTOR_DEFAULT"),
        min_piece_factor_min=default("MIN_PIECE_FACTOR_MIN"),
        min_piece_factor_max=default("MIN_PIECE_FACTOR_MAX"),
        graph_types=tuple(sal_installation.graph_modes),
        routing_strategies=tuple(sal_installation.strategy_labels),
        router_backends=tuple(sal_installation.search_backend_labels),
        heuristic_modes=tuple(sal_installation.heuristic_mode_labels),
        auto_placement_modes=("Manual", "Topological Fields", "Routing-Core Workflow"),
        rotation_modes=("Torque", "Field"),
        room_start_modes=("Room node set", "Centroid terminal"),
        dwelling_source_modes=("Real DB", "Random Synthetic"),
        routing_frame_options=("dominant_walls", "area_inertia_allowed", "area_inertia_rooms"),
        rotation_field_epsilon=1e-6,
        preferred_terminal_marker_size_mm=125.0,
        preferred_terminal_remap_tolerance_mm=300.0,
        history_max_length=400,
        dwelling_export_root=default("DWELLING_EXPORT_ROOT"),
        real_dwelling_db=default("REAL_DWELLING_DB"),
        preferred_shaft_installation=default("PREFERRED_SHAFT_INSTALLATION"),
    )
