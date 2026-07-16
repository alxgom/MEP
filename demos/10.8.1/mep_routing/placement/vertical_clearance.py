"""Vertical-clearance constraints derived from exported cover records."""

from __future__ import annotations

from shapely.geometry import Point
from shapely.ops import unary_union


def insufficient_machine_clearance_regions(rooms, machine_height_mm, clearance_mm=0.0):
    """Return known cover regions whose gross void cannot contain the machine."""
    if machine_height_mm is None:
        return ()
    required_height_mm = float(machine_height_mm) + float(clearance_mm)
    low, adequate, seen = [], [], set()
    for room in rooms:
        for cover in getattr(room, "covers", ()):
            tag = cover.get("tag") or id(cover)
            if tag in seen or cover.get("routing_void_status") != "available":
                continue
            seen.add(tag)
            void_m = cover.get("gross_routing_void_height")
            polygon = cover.get("polygon")
            if void_m is None or polygon is None or polygon.is_empty:
                continue
            (adequate if float(void_m) * 1000.0 + 1e-6 >= required_height_mm else low).append(polygon)
    if not low:
        return ()
    blocked = unary_union(low)
    if adequate:
        blocked = blocked.difference(unary_union(adequate))
    if blocked.is_empty:
        return ()
    return tuple(blocked.geoms) if hasattr(blocked, "geoms") else (blocked,)


def scores_outside_regions(node_scores, nodes, blocked_regions):
    """Keep heatmap scores whose graph nodes are outside machine-only exclusions."""
    if not blocked_regions:
        return node_scores
    return {
        index: score
        for index, score in node_scores.items()
        if index < len(nodes)
        and not any(region.covers(Point(nodes[index][0], nodes[index][1])) for region in blocked_regions)
    }
