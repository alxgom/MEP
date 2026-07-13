from __future__ import annotations

import math


def rotation_field_rooms_for_pin(pin_name, wet_room_names, terminal_names):
    if pin_name in ("left_mid", "right_mid"):
        return [name for name in ("Shaft", "Kitchen") if name in terminal_names or name == "Shaft"]
    return [name for name in wet_room_names if name not in ("Shaft", "Kitchen")]


def rotation_room_weight(room_name, weight_mode_idx):
    if weight_mode_idx == 1:
        return 1.0
    if room_name == "Shaft":
        return 2.0
    if room_name == "Kitchen":
        return 1.5
    return 1.0


def field_alignment_pin_dirs(pin_name, angle, local_axis_to_world_fn):
    local_dirs = {
        "left_mid": [(-1.0, 0.0)],
        "right_mid": [(1.0, 0.0)],
        "tl": [(-1.0, 0.0), (0.0, 1.0)],
        "tr": [(1.0, 0.0), (0.0, 1.0)],
        "bl": [(-1.0, 0.0), (0.0, -1.0)],
        "br": [(1.0, 0.0), (0.0, -1.0)],
    }.get(pin_name, [])
    return [local_axis_to_world_fn(local_dir, angle) for local_dir in local_dirs]


def score_rotation_field_at(
    pins,
    angle,
    wet_room_names,
    terminal_names,
    shaft_point,
    room_target_fn,
    weight_mode_idx,
    local_axis_to_world_fn,
):
    total_score = 0.0
    for pin_name in ("left_mid", "right_mid", "tl", "tr", "bl", "br"):
        pin_pt = pins[pin_name]
        fx = 0.0
        fy = 0.0
        for room_name in rotation_field_rooms_for_pin(pin_name, wet_room_names, terminal_names):
            if room_name == "Shaft":
                if shaft_point is None:
                    continue
                target = shaft_point
            else:
                target = room_target_fn(room_name)
            if target is None:
                continue
            dx = float(target[0]) - float(pin_pt[0])
            dy = float(target[1]) - float(pin_pt[1])
            dist = math.hypot(dx, dy)
            if dist <= 1e-7:
                continue
            w = rotation_room_weight(room_name, weight_mode_idx) / max(250.0, dist)
            fx += w * dx / dist
            fy += w * dy / dist
        field_mag = math.hypot(fx, fy)
        if field_mag <= 1e-12:
            continue
        field_x = fx / field_mag
        field_y = fy / field_mag
        best_pin_alignment = 0.0
        for dir_x, dir_y in field_alignment_pin_dirs(pin_name, angle, local_axis_to_world_fn):
            best_pin_alignment = max(best_pin_alignment, dir_x * field_x + dir_y * field_y)
        total_score += max(0.0, best_pin_alignment) * field_mag
    return total_score


def select_field_alignment_rotation(current_angle, is_valid_rotation_fn, score_rotation_fn, eps):
    current_class = "H" if int(current_angle) % 180 == 0 else "V"
    candidate_angles = {
        "H": [int(current_angle)] if current_class == "H" else [],
        "V": [int(current_angle)] if current_class == "V" else [],
    }
    candidate_angles["H"].extend([0, 180])
    candidate_angles["V"].extend([90, 270])

    scores = {}
    selected_angles = {}
    for orient, angles in candidate_angles.items():
        best_score = None
        best_angle = None
        seen = set()
        for angle in angles:
            angle = int(angle) % 360
            if angle in seen:
                continue
            seen.add(angle)
            if not is_valid_rotation_fn(angle):
                continue
            score = score_rotation_fn(angle)
            if best_score is None or score > best_score:
                best_score = score
                best_angle = angle
        scores[orient] = float(best_score) if best_score is not None else float("-inf")
        selected_angles[orient] = best_angle

    best_orient = max(scores, key=lambda key: scores[key])
    selected = current_class
    new_angle = int(current_angle) % 360
    current_score = scores.get(current_class, float("-inf"))
    if selected_angles.get(best_orient) is not None and scores[best_orient] > current_score + eps:
        selected = best_orient
        new_angle = selected_angles[best_orient]
    return new_angle, selected, scores
