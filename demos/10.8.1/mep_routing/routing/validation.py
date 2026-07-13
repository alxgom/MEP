from __future__ import annotations

from shapely.geometry import LineString


def count_segments_outside_allowed_region(
    routes,
    allowed_region,
    shaft_extraction=None,
    shaft_route_name="Shaft",
    shaft_extraction_tolerance_mm=1.0,
):
    """Count route segments outside the allowed region, excluding shaft-entry geometry."""
    if allowed_region is None:
        return 0

    outside_count = 0
    for route_name, segments in routes:
        for p1, p2 in segments:
            segment = LineString([(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))])
            if (
                route_name == shaft_route_name
                and shaft_extraction is not None
                and segment.distance(shaft_extraction) < shaft_extraction_tolerance_mm
            ):
                continue
            if not segment.covered_by(allowed_region):
                outside_count += 1
    return outside_count


def append_allowed_region_warning(warnings, routes, allowed_region, shaft_extraction=None):
    """Append the standard allowed-region warning when route geometry leaves the routing area."""
    result = list(warnings)
    outside_count = count_segments_outside_allowed_region(routes, allowed_region, shaft_extraction)
    if outside_count:
        result.append(f"{outside_count} segment(s) outside allowed")
    return result
