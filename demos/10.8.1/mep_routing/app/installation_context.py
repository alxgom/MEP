"""Neutral geometry projection from earlier installation results."""

from __future__ import annotations

from shapely.geometry import LineString, Polygon


def installation_context_obstacles(results, *, route_clearance_mm=75.0):
    """Convert completed installation machines and routes into ordered obstacles."""
    obstacles = []
    for result in results:
        pins = getattr(result, "machine_pins", {}) or {}
        corners = [pins.get(name) for name in ("c_tl", "c_tr", "c_br", "c_bl")]
        if all(point is not None for point in corners):
            obstacles.append(Polygon(corners))
        for _route_name, segments in (getattr(result, "routes", ()) or ()):
            for start, end in segments:
                line = LineString([start, end])
                if line.length > 1e-7:
                    obstacles.append(line.buffer(route_clearance_mm, cap_style=2, join_style=2))
    return tuple(obstacles)
