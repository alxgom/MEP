from __future__ import annotations

from shapely.geometry import LineString, Point


def find_route_hit_at_point(routes, world_pt, hit_radius_mm):
    if not routes:
        return None
    click_pt = Point(float(world_pt[0]), float(world_pt[1]))
    best_name = None
    best_dist = float(hit_radius_mm)
    for name, segs in routes:
        for p1, p2 in segs:
            dist = LineString([p1, p2]).distance(click_pt)
            if dist <= best_dist:
                best_dist = dist
                best_name = name
    return (best_name, best_dist) if best_name else None


def find_route_at_point(routes, world_pt, hit_radius_mm):
    hit = find_route_hit_at_point(routes, world_pt, hit_radius_mm)
    return hit[0] if hit else None
