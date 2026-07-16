from __future__ import annotations

import math

from shapely.geometry import Point, Polygon


def routing_frame_axes():
    return ((1.0, 0.0), (0.0, 1.0))


def candidate_room_points(room, axes=None, translation=100.0, placeable_region=None):
    axes = axes or routing_frame_axes()
    candidate_region = room.polygon
    if placeable_region is not None:
        candidate_region = candidate_region.intersection(placeable_region)
    if candidate_region.is_empty:
        return []
    point = candidate_region.representative_point()
    centroid = candidate_region.centroid
    if candidate_region.contains(centroid):
        point = centroid
    base = (round(point.x), round(point.y))
    points = [base]
    for ax, ay in axes:
        points.append((round(base[0] + ax * translation), round(base[1] + ay * translation)))
        points.append((round(base[0] - ax * translation), round(base[1] - ay * translation)))
    seen = set()
    result = []
    for pt in points:
        if pt in seen:
            continue
        if placeable_region is not None and not placeable_region.covers(Point(pt)):
            continue
        seen.add(pt)
        result.append(pt)
    return result


def machine_polygon_from_pins(pins):
    return Polygon([pins["c_tl"], pins["c_tr"], pins["c_br"], pins["c_bl"]])


def area_out_percentage(poly, room_poly):
    if poly.is_empty or room_poly.is_empty:
        return 100.0
    inside = poly.intersection(room_poly).area
    if poly.area <= 1e-7:
        return 100.0
    return max(0.0, (1.0 - inside / poly.area) * 100.0)


def point_angle_to_target(origin, direction, target):
    vx = float(target[0] - origin[0])
    vy = float(target[1] - origin[1])
    norm = math.hypot(vx, vy)
    if norm <= 1e-7:
        return 0.0
    tx, ty = vx / norm, vy / norm
    dx, dy = direction
    dot = max(-1.0, min(1.0, dx * tx + dy * ty))
    cross = dx * ty - dy * tx
    return math.degrees(math.atan2(cross, dot))


def core_like_machine_candidate_score(
    cx,
    cy,
    angle,
    room_poly,
    pins,
    shaft_point,
    kitchen_point,
    include_kitchen_distance,
    boundary_distance_fn,
    local_axis_to_world_fn,
):
    machine_poly = machine_polygon_from_pins(pins)

    large_pin_options = []
    for shaft_pin, kitchen_pin in (("left_mid", "right_mid"), ("right_mid", "left_mid")):
        shaft_dir = local_axis_to_world_fn((-1.0, 0.0) if shaft_pin == "left_mid" else (1.0, 0.0), angle)
        kitchen_dir = local_axis_to_world_fn((-1.0, 0.0) if kitchen_pin == "left_mid" else (1.0, 0.0), angle)
        shaft_angle = abs(point_angle_to_target(pins[shaft_pin], shaft_dir, shaft_point))
        kitchen_angle = abs(point_angle_to_target(pins[kitchen_pin], kitchen_dir, kitchen_point))
        large_pin_options.append((shaft_angle + kitchen_angle, shaft_angle, kitchen_angle, shaft_pin, kitchen_pin))
    _, shaft_angle, kitchen_angle, shaft_pin, kitchen_pin = min(large_pin_options, key=lambda item: item[0])

    out_pct = area_out_percentage(machine_poly, room_poly)
    shaft_clear = boundary_distance_fn(pins[shaft_pin])
    kitchen_clear = boundary_distance_fn(pins[kitchen_pin])
    distance_to_targets = math.hypot(cx - shaft_point[0], cy - shaft_point[1])
    if include_kitchen_distance:
        distance_to_targets += 0.35 * math.hypot(cx - kitchen_point[0], cy - kitchen_point[1])

    return (
        out_pct,
        shaft_angle + kitchen_angle,
        shaft_angle,
        kitchen_angle,
        -shaft_clear,
        -kitchen_clear,
        distance_to_targets,
    )
