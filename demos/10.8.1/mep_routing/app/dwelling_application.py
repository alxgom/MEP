"""Prepare and activate dwellings without interactive-module global state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mep_routing.graphs import GraphLifecycle

from .dwelling_session import ActiveDwellingSession


@dataclass(frozen=True)
class CommonGraphConfig:
    grid_spacing_mm: float
    scaffold_spacing_mm: float
    wall_thickness_mm: float
    wall_clearance_mm: float
    epsilon_mm: float
    medial_sample_spacing_mm: float = 200.0
    cell_boundary_resolution_mm: float = 200.0


@dataclass(frozen=True)
class InstallationGraphProfile:
    port_access_specs: Callable
    simple_input_builder: Callable | None = None
    simple_axis_clearance_mm: float = 100.0


@dataclass(frozen=True)
class TerminalRuntimeConfig:
    regulation_clearance_mm: float
    terminal_buffer_mm: float
    remap_tolerance_mm: float


def create_active_dwelling(prepared, graph: CommonGraphConfig, profile: InstallationGraphProfile, terminals: TerminalRuntimeConfig):
    """Create a fresh dwelling aggregate using an installation graph profile."""
    lifecycle = GraphLifecycle(
        routing_region=prepared.routing_region_base,
        wall_polygons=prepared.wall_polygons,
        covers=prepared.covers,
        columns=prepared.columns,
        shafts=prepared.shafts,
        walls=prepared.walls,
        terminals=prepared.terminals,
        shaft_extraction=prepared.shaft_extraction,
        shaft_core_entry_specs=prepared.shaft_core_entry_specs,
        grid_spacing_mm=graph.grid_spacing_mm,
        scaffold_spacing_mm=graph.scaffold_spacing_mm,
        wall_thickness_mm=graph.wall_thickness_mm,
        wall_clearance_mm=graph.wall_clearance_mm,
        epsilon_mm=graph.epsilon_mm,
        port_access_specs=profile.port_access_specs,
        simple_input_builder=profile.simple_input_builder,
        simple_axis_clearance_mm=profile.simple_axis_clearance_mm,
        medial_sample_spacing_mm=graph.medial_sample_spacing_mm,
        cell_boundary_resolution_mm=graph.cell_boundary_resolution_mm,
    )
    return ActiveDwellingSession.create(
        prepared, lifecycle,
        regulation_clearance_mm=terminals.regulation_clearance_mm,
        terminal_buffer_mm=terminals.terminal_buffer_mm,
        remap_tolerance_mm=terminals.remap_tolerance_mm,
    )


def load_real_dwelling(*, loader: Callable, summarizer: Callable | None, db_path: Path, case: Any, frame_name: str, preferred_shaft_installation: str):
    """Load one selected dwelling-export case through injected adapters."""
    if loader is None:
        raise RuntimeError("dwelling_export.demo_loader could not be imported.")
    if not db_path.exists():
        raise RuntimeError(f"Dwelling database not found: {db_path}")
    scenario = loader(
        db_path=db_path,
        execution=case.execution,
        dwelling_id=case.dwelling_id,
        project_guid=case.project_guid,
        scale_to_mm=True,
        frame_name=frame_name,
        preferred_shaft_installation=preferred_shaft_installation,
    )
    return scenario, case.label, summarizer(scenario) if summarizer else {}
