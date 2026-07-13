"""Runtime graph assembly independent from the interactive application state.

Graph builders produce nodes and directed edge records.  This module turns
those records into the adjacency, edge-coordinate, and spatial-index objects
consumed by the interactive router.  App-specific geometry and cache
invalidation stay in the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np
from scipy.spatial import cKDTree

from .env import EnvView


REVERSE_DIRECTION = {"E": "W", "N": "S", "W": "E", "S": "N"}

Edge = tuple[int, int, float, str]


@dataclass(frozen=True)
class RuntimeGraph:
    """The runtime representation shared by routing, drawing, and snapping."""

    nodes: np.ndarray
    adjacency: dict[int, list[tuple[int, float, str]]]
    edge_list: list[Edge]
    edge_coords: np.ndarray
    spatial_index: object
    env: EnvView


def append_shaft_runtime_node(
    nodes: np.ndarray,
    edge_list: Iterable[Edge],
    *,
    shaft_center: tuple[float, float] | None,
    shaft_bounds: tuple[float, float, float, float] | None,
    clearance_mm: float,
    face_tolerance_mm: float = 10.0,
) -> tuple[np.ndarray, list[Edge]]:
    """Append a shaft-center node and join it to matching shaft-face nodes.

    If no generated node lies on a clearance face, the closest node supplies
    one fallback connection.  ``None`` shaft inputs leave the graph unchanged.
    """
    nodes = np.asarray(nodes)
    edges = list(edge_list)
    if shaft_center is None or shaft_bounds is None or not len(nodes):
        return nodes, edges

    center_x, center_y = shaft_center
    min_x, min_y, max_x, max_y = shaft_bounds
    shaft_index = len(nodes)
    result_nodes = np.vstack([nodes, [round(center_x), round(center_y)]])
    face_points = (
        (max_x + clearance_mm, (min_y + max_y) / 2, "W"),
        (min_x - clearance_mm, (min_y + max_y) / 2, "E"),
        ((min_x + max_x) / 2, max_y + clearance_mm, "S"),
        ((min_x + max_x) / 2, min_y - clearance_mm, "N"),
    )

    connected = False
    for point_x, point_y, direction in face_points:
        distances = np.hypot(nodes[:, 0] - point_x, nodes[:, 1] - point_y)
        nearest_index = int(np.argmin(distances))
        if distances[nearest_index] < face_tolerance_mm:
            length = float(np.hypot(center_x - nodes[nearest_index, 0], center_y - nodes[nearest_index, 1]))
            edges.append((nearest_index, shaft_index, length, direction))
            connected = True

    if not connected:
        distances = np.hypot(nodes[:, 0] - center_x, nodes[:, 1] - center_y)
        nearest_index = int(np.argmin(distances))
        delta_x = center_x - nodes[nearest_index, 0]
        delta_y = center_y - nodes[nearest_index, 1]
        if abs(delta_x) > abs(delta_y):
            direction = "E" if delta_x > 0 else "W"
        else:
            direction = "N" if delta_y > 0 else "S"
        edges.append((nearest_index, shaft_index, float(np.hypot(delta_x, delta_y)), direction))

    return result_nodes, edges


def restrict_pin_access_edges(
    edge_list: Iterable[Edge],
    pin_indices: Mapping[int, str],
    allowed_directions_by_pin: Mapping[str, set[str]],
) -> list[Edge]:
    """Remove edges that leave a snapped machine pin through a forbidden side."""
    filtered_edges: list[Edge] = []
    for source, target, length, direction in edge_list:
        source_pin = pin_indices.get(source)
        target_pin = pin_indices.get(target)
        source_allowed = allowed_directions_by_pin.get(source_pin, set()) if source_pin else set()
        target_allowed = allowed_directions_by_pin.get(target_pin, set()) if target_pin else set()
        if source_allowed and direction not in source_allowed:
            continue
        if target_allowed and REVERSE_DIRECTION[direction] not in target_allowed:
            continue
        filtered_edges.append((source, target, length, direction))
    return filtered_edges


def build_adjacency(node_count: int, edge_list: Iterable[Edge]) -> dict[int, list[tuple[int, float, str]]]:
    """Build the bidirectional directed adjacency used by routing searches."""
    adjacency = {index: [] for index in range(node_count)}
    for source, target, length, direction in edge_list:
        adjacency[source].append((target, length, direction))
        adjacency[target].append((source, length, REVERSE_DIRECTION[direction]))
    return adjacency


def edge_coordinates(nodes: np.ndarray, edge_list: Iterable[Edge]) -> np.ndarray:
    """Return screen/distance-friendly endpoint rows for graph edges."""
    edges = list(edge_list)
    if not edges:
        return np.empty((0, 4), dtype=np.float32)
    return np.array(
        [[nodes[source, 0], nodes[source, 1], nodes[target, 0], nodes[target, 1]] for source, target, _length, _direction in edges],
        dtype=np.float32,
    )


def create_runtime_graph(
    nodes: np.ndarray,
    edge_list: Iterable[Edge],
    *,
    spatial_index_factory: Callable[[np.ndarray], object] = cKDTree,
) -> RuntimeGraph:
    """Materialize graph runtime data after node/edge construction is complete."""
    runtime_nodes = np.asarray(nodes, dtype=np.float32)
    runtime_edges = list(edge_list)
    adjacency = build_adjacency(len(runtime_nodes), runtime_edges)
    return RuntimeGraph(
        nodes=runtime_nodes,
        adjacency=adjacency,
        edge_list=runtime_edges,
        edge_coords=edge_coordinates(runtime_nodes, runtime_edges),
        spatial_index=spatial_index_factory(runtime_nodes),
        env=EnvView(runtime_nodes, adjacency),
    )
