"""Application facade for graph construction and runtime synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from mep_routing.graphs import GraphMode


@dataclass
class GraphBuildSession:
    workspace: Any
    terminal_runtime: Any
    invalidate_candidates: Callable[[], None]
    invalidate_validity: Callable[[], None]
    report: Callable[[str], None] = print

    def build_base_regular(self):
        started = perf_counter()
        runtime = self.workspace.build_base_regular()
        if runtime is not None:
            self.report(f"[Base Regular Grid] Built {len(runtime.nodes)} nodes in {(perf_counter() - started) * 1000.0:.1f}ms")
        return runtime

    def build(self, mode, pins, angle, *, shift_hannan_walls=True):
        result = self.workspace.build_selected(
            int(mode), pins, angle,
            shift_hannan_walls=shift_hannan_walls,
            terminal_runtime=self.terminal_runtime,
        )
        if result is None:
            return None
        self.invalidate_candidates()
        self.invalidate_validity()
        self._report_result(GraphMode(int(mode)), result)
        return result

    def apply_machine_obstacle(self, polygon, pins, angle, *, clearance_mm):
        if not self.workspace.grid_available:
            result = self.workspace.apply_machine_obstacle(
                polygon, {}, angle, clearance_mm=clearance_mm,
                terminal_runtime=self.terminal_runtime,
            )
            self.invalidate_candidates()
            return result
        started = perf_counter()
        result = self.workspace.apply_machine_obstacle(
            polygon, pins, angle, clearance_mm=clearance_mm,
            terminal_runtime=self.terminal_runtime,
        )
        self.invalidate_candidates()
        self.report(
            f"Grid update: {(perf_counter() - started) * 1000.0:.1f} ms  "
            f"(blocked nodes={result.blocked_node_count}, edges={result.blocked_edge_count})"
        )
        return result

    def _report_result(self, mode: GraphMode, result) -> None:
        variant = result.variant
        if variant is None or not len(variant.nodes):
            return
        if mode is GraphMode.HANNAN_MEDIAL:
            info = variant.diagnostics
            self.report(
                f"[Hanan + Voronoi] skeleton nodes={info['skeleton_nodes']} "
                f"local links={info['links']} axes={len(variant.axes_x)}x{len(variant.axes_y)} "
                f"nodes={len(variant.nodes)} edges={len(variant.edges)} in {result.elapsed_ms:.1f}ms "
                "(shared junctions; Euclidean heuristic; fixed bends)"
            )
        elif mode is GraphMode.HANNAN:
            self.report(
                f"[Hannan Simple] axes={len(variant.axes_x)}x{len(variant.axes_y)} "
                f"nodes={len(variant.nodes)} edges={len(variant.edges)} in {result.elapsed_ms:.1f}ms "
                f"(axes {variant.axes_ms:.1f}, nodes {variant.nodes_ms:.1f}, edges {variant.edges_ms:.1f})"
            )
        elif mode is GraphMode.EPSILON:
            self.report(
                f"[Epsilon Core-like] eps={self.workspace.lifecycle.epsilon_mm:.0f} "
                f"axes={len(variant.axes_x)}x{len(variant.axes_y)} nodes={len(variant.nodes)} "
                f"edges={len(variant.edges)} in {result.elapsed_ms:.1f}ms "
                f"(axes {variant.axes_ms:.1f}, nodes {variant.nodes_ms:.1f}, edges {variant.edges_ms:.1f})"
            )
        elif mode is GraphMode.CORE_SIMPLE:
            self.report(
                f"[Routing-Core Simple Port] axes={len(variant.axes_x)}x{len(variant.axes_y)} "
                f"nodes={len(variant.nodes)} edges={len(variant.edges)} in {result.elapsed_ms:.1f}ms"
            )
        elif mode is GraphMode.MEDIAL_AXIS:
            self.report(
                f"[Medial Axis] spacing={variant.diagnostics['sample_spacing_mm']:.0f}mm "
                f"samples={variant.diagnostics['samples']} nodes={len(variant.nodes)} "
                f"edges={len(variant.edges)} in {result.elapsed_ms:.1f}ms "
                "(Euclidean heuristic; fixed bends; experimental diagonal metrics)"
            )
        elif mode is GraphMode.CELL_SKELETON:
            info = variant.diagnostics
            self.report(
                f"[Cell Skeleton] cells={len(info['cells'])} portals={info['portals']} "
                f"nodes={len(variant.nodes)} edges={len(variant.edges)} in {result.elapsed_ms:.1f}ms "
                f"(attached={info['attached']}/{len(variant.required_points)}, "
                f"excluded boundary area={info['excluded_area_mm2']:.0f}mm2)"
            )
