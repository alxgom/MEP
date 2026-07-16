"""Normalize dwelling-export source coordinates for the routing workbench."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import numpy as np
from shapely.geometry import LineString, Point


def transform_source_point(point_xy, summary: Mapping[str, Any] | None, *, scale_to_mm: float):
    """Transform one exported source point into the demo's millimetre frame."""
    if not point_xy or summary is None:
        return None
    x, y = float(point_xy[0]), float(point_xy[1])
    frame = summary.get("routing_frame") or {}
    metadata = summary.get("metadata") or {}
    rotation_degrees = float(frame.get("rotation_degrees", 0.0) or 0.0)
    origin = metadata.get("source_rotation_origin", [0.0, 0.0])
    bounds_origin = metadata.get("bounds_origin", [0.0, 0.0])
    if abs(rotation_degrees) > 1e-9:
        theta = math.radians(-rotation_degrees)
        ox, oy = float(origin[0]), float(origin[1])
        dx, dy = x - ox, y - oy
        x = ox + dx * math.cos(theta) - dy * math.sin(theta)
        y = oy + dx * math.sin(theta) + dy * math.cos(theta)
    return (
        round((x - float(bounds_origin[0])) * scale_to_mm),
        round((y - float(bounds_origin[1])) * scale_to_mm),
    )


def build_core_shaft_entry_specs(
    scenario,
    *,
    summarize: Callable[[Any], Mapping[str, Any]] | None,
    scale_to_mm: float,
    match_tolerance_mm: float,
) -> list[dict[str, Any]]:
    """Extract routing-core shaft entries in the normalized demo frame."""
    summary = summarize(scenario) if summarize else {"metadata": getattr(scenario, "metadata", {})}
    raw_meta = summary.get("metadata") or {}
    candidates = raw_meta.get("shaft_candidates") or []
    candidate_idx = raw_meta.get("shaft_candidate_index")
    if candidate_idx is None or candidate_idx < 0 or candidate_idx >= len(candidates):
        candidates = [candidate for candidate in candidates if candidate.get("candidate_for_living")]
    else:
        candidates = [candidates[candidate_idx]]

    specs = []
    for candidate in candidates:
        for routing in candidate.get("routing") or []:
            centroid = transform_source_point(routing.get("centroid"), summary, scale_to_mm=scale_to_mm)
            if centroid is None:
                continue
            exit_walls = []
            for wall in routing.get("exit_walls") or []:
                if len(wall) < 2:
                    continue
                p1 = transform_source_point(wall[0], summary, scale_to_mm=scale_to_mm)
                p2 = transform_source_point(wall[1], summary, scale_to_mm=scale_to_mm)
                if p1 is not None and p2 is not None:
                    exit_walls.append((p1, p2))
            entry_points = routing.get("entry_points") or []
            if not entry_points and routing.get("entry_point"):
                entry_points = [routing.get("entry_point")]
            for entry_raw in entry_points:
                entry = transform_source_point(entry_raw, summary, scale_to_mm=scale_to_mm)
                if entry is None:
                    continue
                normal = np.array([entry[0] - centroid[0], entry[1] - centroid[1]], dtype=np.float64)
                norm = float(np.linalg.norm(normal))
                if norm < 1e-6:
                    continue
                normal /= norm
                entry_point = Point(float(entry[0]), float(entry[1]))
                matching_wall = min(exit_walls, key=lambda wall: LineString(wall).distance(entry_point)) if exit_walls else None
                if matching_wall is not None and LineString(matching_wall).distance(entry_point) > match_tolerance_mm:
                    matching_wall = None
                specs.append({
                    "centroid": (float(centroid[0]), float(centroid[1])),
                    "entry": (float(entry[0]), float(entry[1])),
                    "normal": (float(normal[0]), float(normal[1])),
                    "exit_wall": matching_wall,
                    "id": routing.get("id"),
                })
    return specs
