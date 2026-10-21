"""Graph lifecycle coordination independent from interactive application state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from shapely.geometry import Polygon

from mep_routing.geometry.directions import travel_direction

from .dynamic import filter_dynamic_machine_obstacle
from .hannan import build_static_axes
from .regular import build_regular_grid
from .runtime import RuntimeGraph, append_shaft_runtime_node, create_runtime_graph, restrict_pin_access_edges
from .variants import GraphVariantResult, build_epsilon_variant, build_hannan_variant, build_simple_variant, build_medial_variant
from .variants import build_cell_variant


class GraphMode(IntEnum):
    REGULAR = 0
    HANNAN = 1
    EPSILON = 2
    CORE_SIMPLE = 3
    MEDIAL_AXIS = 4
    CELL_SKELETON = 5


@dataclass(frozen=True)
class GraphBuildResult:
    """Uncommitted graph result and optional variant diagnostics."""

    runtime: RuntimeGraph
    graph_type: int
    elapsed_ms: float
    variant: GraphVariantResult | None = None


@dataclass(frozen=True)
class DynamicGraphResult:
    """Active graph view after applying the machine obstacle."""

    env: Any
    blocked_node_count: int
    blocked_edge_count: int


@dataclass
class GraphLifecycle:
    """Build base and active graph modes from explicit dwelling and machine inputs."""

    routing_region: Any
    wall_polygons: Sequence[Any]
    covers: Sequence[Any]
    columns: Sequence[Any]
    shafts: Sequence[Any]
    walls: Sequence[Any]
    terminals: Mapping[str, tuple[float, float]]
    shaft_extraction: Any
    shaft_core_entry_specs: Sequence[dict[str, Any]]
    grid_spacing_mm: float
    scaffold_spacing_mm: float
    wall_thickness_mm: float
    wall_clearance_mm: float
    epsilon_mm: float
    port_access_specs: Callable[[Mapping[str, Any], float], Sequence[Mapping[str, Any]]]
    simple_input_builder: Callable[..., Any] | None = None
    simple_axis_clearance_mm: float = 100.0
    medial_sample_spacing_mm: float = 200.0
    cell_boundary_resolution_mm: float = 200.0
    _hannan_templates: dict[bool, Mapping[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def node_region(self):
        """Return the routing region inset by the configured wall clearance."""
        if self.routing_region is None:
            return None
        if self.wall_clearance_mm <= 0:
            return self.routing_region
        inset = self.routing_region.buffer(-self.wall_clearance_mm, join_style=2)
        return self.routing_region if inset.is_empty else inset

    def clear_hannan_templates(self) -> None:
        """Discard static Hannan axes after a dwelling geometry change."""
        self._hannan_templates.clear()

    def hannan_template(self, *, shift_walls: bool) -> Mapping[str, Any]:
        """Return the cached static Hannan axes for the requested wall policy."""
        cache_key = bool(shift_walls)
        if cache_key not in self._hannan_templates:
            self._hannan_templates[cache_key] = build_static_axes(
                allowed_region=self.routing_region,
                terminals=self.terminals,
                shaft_extraction=self.shaft_extraction,
                covers=self.covers,
                columns=self.columns,
                shafts=self.shafts,
                wall_polys=self.wall_polygons,
                walls=self.walls,
                grid_spacing_mm=self.grid_spacing_mm,
                scaffold_spacing_mm=self.scaffold_spacing_mm,
                wall_clearance_mm=self.wall_clearance_mm,
                shift_walls=shift_walls,
            )
        return self._hannan_templates[cache_key]

    def build_base_regular(self) -> RuntimeGraph | None:
        """Build the unmodified regular graph used by placement fields."""
        if self.routing_region is None:
            return None
        nodes, edges = build_regular_grid(
            self.routing_region,
            self.node_region(),
            self.wall_polygons,
            self.grid_spacing_mm,
            self.wall_thickness_mm,
        )
        return create_runtime_graph(nodes, edges)

    def commit_runtime(self, nodes, edges, machine_pins, machine_angle: float, *, euclidean=False, exact_access=False, cells=()) -> RuntimeGraph:
        """Attach shaft/pin constraints and materialize a routing runtime graph."""
        shaft_center = shaft_bounds = None
        if self.shaft_extraction is not None:
            representative = self.shaft_extraction.representative_point()
            shaft_center = (round(float(representative.x)), round(float(representative.y)))
            shaft_bounds = self.shaft_extraction.bounds
        nodes, edges = append_shaft_runtime_node(
            nodes,
            edges,
            shaft_center=shaft_center,
            shaft_bounds=shaft_bounds,
            clearance_mm=self.wall_clearance_mm,
        )
        if euclidean:
            edges = [(u, v, length, travel_direction(nodes[u], nodes[v])) for u, v, length, _ in edges]

        pin_indices: dict[int, str] = {}
        if len(nodes):
            for name, point in machine_pins.items():
                distances = np.hypot(nodes[:, 0] - point[0], nodes[:, 1] - point[1])
                index = int(np.argmin(distances))
                if distances[index] < 1.0:
                    pin_indices[index] = name
        allowed_directions: dict[str, set[str]] = {}
        for spec in self.port_access_specs(machine_pins, machine_angle):
            allowed_directions.setdefault(spec["pin"], set()).add(spec["out_dir"])
        return create_runtime_graph(
            nodes,
            restrict_pin_access_edges(edges, pin_indices, allowed_directions),
            euclidean=euclidean,
            exact_access=exact_access, cells=cells,
        )

    def build_selected(
        self,
        graph_type: int,
        machine_pins,
        machine_angle: float,
        *,
        shift_hannan_walls: bool = True,
    ) -> GraphBuildResult | None:
        """Build a supported routing graph, accepting its stable integer enum value."""
        if self.routing_region is None:
            return None
        try:
            mode = GraphMode(graph_type)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported graph mode: {graph_type!r}") from exc
        started = perf_counter()
        if mode is GraphMode.REGULAR:
            nodes, edges = build_regular_grid(
                self.routing_region,
                self.node_region(),
                self.wall_polygons,
                self.grid_spacing_mm,
                self.wall_thickness_mm,
            )
            runtime = self.commit_runtime(nodes, edges, machine_pins, machine_angle)
            return GraphBuildResult(runtime, int(mode), (perf_counter() - started) * 1000.0)

        access_points = [spec["access_point"] for spec in self.port_access_specs(machine_pins, machine_angle)]
        if mode is GraphMode.HANNAN:
            variant = build_hannan_variant(
                template=self.hannan_template(shift_walls=shift_hannan_walls),
                allowed_region=self.routing_region,
                node_region=self.node_region(),
                wall_polys=self.wall_polygons,
                wall_thickness_mm=self.wall_thickness_mm,
                terminals=self.terminals,
                shaft_extraction=self.shaft_extraction,
                machine_access_points=access_points,
            )
        elif mode is GraphMode.EPSILON:
            variant = build_epsilon_variant(
                allowed_region=self.routing_region,
                node_region=self.node_region(),
                covers=self.covers,
                columns=self.columns,
                shafts=self.shafts,
                wall_polys=self.wall_polygons,
                wall_thickness_mm=self.wall_thickness_mm,
                terminals=self.terminals,
                shaft_core_entry_specs=self.shaft_core_entry_specs,
                shaft_extraction=self.shaft_extraction,
                machine_access_points=access_points,
                epsilon_mm=self.epsilon_mm,
                scaffold_spacing_mm=self.scaffold_spacing_mm,
            )
        elif mode in (GraphMode.MEDIAL_AXIS, GraphMode.CELL_SKELETON):
            required = list(self.terminals.values()) + access_points
            # Preserve the existing explicit shaft-face entry semantics.
            if self.shaft_extraction is not None:
                min_x, min_y, max_x, max_y = self.shaft_extraction.bounds
                c = self.wall_clearance_mm
                required.extend(((max_x + c, (min_y + max_y) / 2), (min_x - c, (min_y + max_y) / 2),
                                 ((min_x + max_x) / 2, max_y + c), ((min_x + max_x) / 2, min_y - c)))
            corners = [machine_pins.get(name) for name in ("c_tl", "c_tr", "c_br", "c_bl")]
            machine_polygon = Polygon(corners) if all(p is not None for p in corners) else None
            builder = build_medial_variant if mode is GraphMode.MEDIAL_AXIS else build_cell_variant
            options = {"sample_spacing_mm": self.medial_sample_spacing_mm} if mode is GraphMode.MEDIAL_AXIS else {
                "boundary_resolution_mm": self.cell_boundary_resolution_mm,
            }
            variant = builder(
                allowed_region=self.node_region(), wall_polys=self.wall_polygons,
                wall_thickness_mm=self.wall_thickness_mm, required_points=required,
                machine_polygon=machine_polygon, **options,
            )
        else:
            if self.simple_input_builder is None:
                raise ValueError("core-simple graph mode has no installation input builder")
            variant = build_simple_variant(self.simple_input_builder(
                routing_region=self.routing_region,
                terminals=self.terminals,
                columns=self.columns,
                shafts=self.shafts,
                machine_pins=machine_pins,
                access_points=access_points,
                epsilon_mm=self.epsilon_mm,
                axis_clearance_mm=self.simple_axis_clearance_mm,
            ))
        runtime = self.commit_runtime(variant.nodes, variant.edges, machine_pins, machine_angle,
                                      euclidean=mode is GraphMode.MEDIAL_AXIS,
                                      exact_access=mode in (GraphMode.MEDIAL_AXIS, GraphMode.CELL_SKELETON),
                                      cells=variant.diagnostics.get("cells", ()))
        return GraphBuildResult(runtime, int(mode), (perf_counter() - started) * 1000.0, variant)

    def apply_dynamic_obstacle(
        self,
        runtime: RuntimeGraph,
        machine_polygon,
        machine_pins,
        machine_angle: float,
        *,
        clearance_mm: float,
    ):
        """Filter a committed graph while preserving terminal and port-access nodes."""
        if not len(runtime.nodes):
            return DynamicGraphResult(runtime.env, 0, 0)
        protected_nodes = set()
        protected_points = list(self.terminals.values())
        protected_points.extend(spec["access_point"] for spec in self.port_access_specs(machine_pins, machine_angle))
        for point in protected_points:
            _distance, index = runtime.spatial_index.query(point)
            index = int(index)
            if np.hypot(runtime.nodes[index, 0] - point[0], runtime.nodes[index, 1] - point[1]) < 1.0:
                protected_nodes.add(index)
        env, blocked_nodes, blocked_edges = filter_dynamic_machine_obstacle(
            runtime.nodes,
            runtime.edge_list,
            runtime.edge_coords,
            machine_polygon,
            clearance_mm,
            protected_nodes,
            euclidean=runtime.env.euclidean,
            exact_access=runtime.env.exact_access, cells=runtime.env.cells,
        )
        return DynamicGraphResult(env, blocked_nodes, blocked_edges)
