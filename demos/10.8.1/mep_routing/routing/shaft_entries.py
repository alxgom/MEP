"""Shaft-entry geometry and graph-node selection for routing."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from shapely.geometry import LineString, Point


def shaft_representative_point(shaft_geometry) -> tuple[float, float]:
    """Choose the centroid when usable, otherwise a guaranteed interior point."""
    centroid = shaft_geometry.centroid
    point = centroid if shaft_geometry.contains(centroid) else shaft_geometry.representative_point()
    return float(round(point.x)), float(round(point.y))


def shaft_entry_geometry(shaft_geometry, node: Iterable[float]) -> dict[str, Any]:
    """Describe the radial path from a shaft to one graph node."""
    node_x, node_y = (float(value) for value in node)
    node_point = Point(node_x, node_y)
    rep_x, rep_y = shaft_representative_point(shaft_geometry)
    if shaft_geometry.contains(node_point):
        return {
            "rep": (rep_x, rep_y),
            "entry": (node_x, node_y),
            "node": (node_x, node_y),
            "distance": 0.0,
            "orthogonality_error": 0.0,
        }

    boundary = shaft_geometry.boundary
    entry_point = boundary.interpolate(boundary.project(node_point))
    entry = np.array((float(entry_point.x), float(entry_point.y)))
    rep = np.array((rep_x, rep_y))
    node_array = np.array((node_x, node_y))
    outward = node_array - entry
    radial = entry - rep
    distance = float(np.linalg.norm(outward))
    radial_length = float(np.linalg.norm(radial))
    if distance < 1e-6 or radial_length < 1e-6:
        orthogonality_error = 1.0
    else:
        if float(np.dot(outward, radial)) < 0.0:
            radial = -radial
        alignment = abs(float(np.dot(outward, radial)) / (distance * radial_length))
        orthogonality_error = 1.0 - min(1.0, alignment)
    return {
        "rep": (rep_x, rep_y),
        "entry": (float(entry[0]), float(entry[1])),
        "node": (node_x, node_y),
        "distance": distance,
        "orthogonality_error": orthogonality_error,
    }


def select_shaft_entry_nodes(
    nodes: np.ndarray,
    shaft_geometry,
    *,
    search_radius_mm: float,
    grid_spacing_mm: float,
    max_candidates: int,
    core_entry_specs: Iterable[dict[str, Any]] = (),
    spatial_index=None,
) -> tuple[list[int], int | None, dict[int, dict[str, Any]]]:
    """Rank graph nodes that can serve as shaft entries.

    Routing-core entry specs take precedence when supplied; otherwise the
    closest outward, radial shaft access nodes are selected.
    """
    nodes = np.asarray(nodes)
    if shaft_geometry is None or not len(nodes):
        return [], None, {}

    geometries: dict[int, dict[str, Any]] = {}
    core_candidates = []
    for spec_index, spec in enumerate(core_entry_specs):
        entry = np.asarray(spec["entry"], dtype=np.float64)
        centroid = np.asarray(spec["centroid"], dtype=np.float64)
        normal = np.asarray(spec["normal"], dtype=np.float64)
        exit_wall = spec.get("exit_wall")
        search_indices = range(len(nodes))
        if spatial_index is not None:
            found = spatial_index.query_ball_point(entry, search_radius_mm)
            if found:
                search_indices = found
        for node_index in search_indices:
            node = np.asarray(nodes[int(node_index)], dtype=np.float64)
            node_point = Point(float(node[0]), float(node[1]))
            if shaft_geometry.contains(node_point):
                continue
            offset = node - entry
            distance = float(np.linalg.norm(offset))
            if distance > search_radius_mm:
                continue
            alignment = 1.0 if distance < 1e-6 else float(np.dot(offset / distance, normal))
            if alignment < -1e-6:
                continue
            orthogonality_error = 1.0 - max(0.0, min(1.0, alignment))
            exit_wall_penalty = 0.0
            if exit_wall is not None:
                exit_wall_penalty = min(search_radius_mm, LineString(exit_wall).distance(Point(float(entry[0]), float(entry[1]))))
            shaft_distance = node_point.distance(shaft_geometry)
            score = distance + orthogonality_error * grid_spacing_mm * 4.0 + shaft_distance * 0.25 + exit_wall_penalty
            geometry = {
                "rep": (float(centroid[0]), float(centroid[1])),
                "entry": (float(entry[0]), float(entry[1])),
                "node": (float(node[0]), float(node[1])),
                "distance": distance,
                "orthogonality_error": orthogonality_error,
                "source": "routing_core",
                "score": score,
                "spec_idx": spec_index,
            }
            old = geometries.get(int(node_index))
            if old is None or score < old["score"]:
                geometries[int(node_index)] = geometry
            core_candidates.append((score, distance, orthogonality_error, int(node_index)))

    if core_candidates:
        core_candidates.sort()
        selected = []
        seen = set()
        for _score, _distance, _error, node_index in core_candidates:
            if node_index not in seen:
                seen.add(node_index)
                selected.append(node_index)
            if len(selected) >= max_candidates:
                break
        return selected, selected[0], geometries

    candidates = []
    for node_index, node in enumerate(nodes):
        node_point = Point(float(node[0]), float(node[1]))
        if shaft_geometry.contains(node_point):
            continue
        distance = node_point.distance(shaft_geometry)
        if distance > search_radius_mm:
            continue
        geometry = shaft_entry_geometry(shaft_geometry, node)
        geometries[int(node_index)] = geometry
        score = distance + geometry["orthogonality_error"] * grid_spacing_mm * 2.0
        candidates.append((score, distance, geometry["orthogonality_error"], int(node_index)))
    if candidates:
        candidates.sort()
        selected = [node_index for _score, _distance, _error, node_index in candidates[:max_candidates]]
        return selected, selected[0], geometries

    rep_x, rep_y = shaft_representative_point(shaft_geometry)
    if spatial_index is not None:
        _distance, node_index = spatial_index.query((rep_x, rep_y))
        return [int(node_index)], int(node_index), geometries
    differences = np.hypot(nodes[:, 0] - rep_x, nodes[:, 1] - rep_y)
    node_index = int(np.argmin(differences))
    return [node_index], node_index, geometries


def shaft_entry_segments(geometry: dict[str, Any] | None, *, min_length_mm: float = 1.0):
    """Return visible route segments from shaft representative point to node."""
    if geometry is None:
        return []
    segments = []
    rep, entry, node = geometry["rep"], geometry["entry"], geometry["node"]
    if float(np.hypot(entry[0] - rep[0], entry[1] - rep[1])) > min_length_mm:
        segments.append((rep, entry))
    if float(np.hypot(node[0] - entry[0], node[1] - entry[1])) > min_length_mm:
        segments.append((entry, node))
    return segments
