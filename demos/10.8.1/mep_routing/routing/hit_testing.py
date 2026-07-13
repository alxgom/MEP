from __future__ import annotations

import math

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


def selected_pin_names(selected_route_name, routes, global_pins, pin_match_distance=2.0):
    if not selected_route_name or not routes or not global_pins:
        return set()
    pin_names = [p for p in ("tl", "tr", "bl", "br", "left_mid", "right_mid") if p in global_pins]
    selected = set()
    for route_name, segs in routes:
        if route_name != selected_route_name:
            continue
        for p1, p2 in segs[-3:]:
            for pin_name in pin_names:
                pin_pt = global_pins[pin_name]
                if math.hypot(float(p1[0]) - pin_pt[0], float(p1[1]) - pin_pt[1]) < pin_match_distance:
                    selected.add(pin_name)
                if math.hypot(float(p2[0]) - pin_pt[0], float(p2[1]) - pin_pt[1]) < pin_match_distance:
                    selected.add(pin_name)
    return selected
