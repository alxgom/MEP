from __future__ import annotations


def pin_nodes_from_pins(global_pins, nearest_node_fn):
    pin_nodes = {}
    for pin_name, pt in global_pins.items():
        if pin_name.startswith("c_"):
            continue
        pin_nodes[pin_name] = int(nearest_node_fn(pt))
    return pin_nodes


def rotation_score_from_fields(pin_nodes, distance_fields, wet_room_names, weights):
    d_left = distance_fields["Shaft"].get(pin_nodes["left_mid"], 1e9)
    d_right = distance_fields["Shaft"].get(pin_nodes["right_mid"], 1e9)
    if d_left < d_right:
        kitchen_pin = "right_mid"
        shaft_dist = d_left
    else:
        kitchen_pin = "left_mid"
        shaft_dist = d_right

    kitchen_dist = 0.0
    if "Kitchen" in distance_fields:
        kitchen_dist = distance_fields["Kitchen"].get(pin_nodes[kitchen_pin], 1e9)

    small_pins = ["tl", "tr", "bl", "br"]
    room_dists = 0.0
    remaining_rooms = [r for r in wet_room_names if r != "Kitchen"]
    used_pins = set()

    for room_name in remaining_rooms:
        best_d = 1e9
        best_p = None
        for pin_name in small_pins:
            if pin_name in used_pins:
                continue
            d = distance_fields[room_name].get(pin_nodes[pin_name], 1e9)
            if d < best_d:
                best_d = d
                best_p = pin_name
        if best_p is not None:
            used_pins.add(best_p)
            room_dists += best_d
        else:
            room_dists += 1e9

    return weights["Shaft"] * shaft_dist + weights["Kitchen"] * kitchen_dist + room_dists


def best_valid_rotation_for_point(
    x,
    y,
    rotations,
    is_valid_fn,
    pins_fn,
    nearest_node_fn,
    distance_fields,
    wet_room_names,
    weights,
):
    best_rot = None
    min_rot_score = 1e18

    for rot in rotations:
        if not is_valid_fn(x, y, rot):
            continue
        pin_nodes = pin_nodes_from_pins(pins_fn(x, y, rot), nearest_node_fn)
        rot_score = rotation_score_from_fields(pin_nodes, distance_fields, wet_room_names, weights)
        if rot_score < min_rot_score:
            min_rot_score = rot_score
            best_rot = rot

    return best_rot, min_rot_score


def choose_topological_machine_placement(
    env,
    node_scores,
    distance_fields,
    rotations,
    is_valid_fn,
    pins_fn,
    nearest_node_fn,
    wet_room_names,
    weights,
):
    for node_idx in sorted(node_scores.keys(), key=lambda n: node_scores[n]):
        x, y = env.nodes[node_idx][0], env.nodes[node_idx][1]
        best_rot, score = best_valid_rotation_for_point(
            x,
            y,
            rotations,
            is_valid_fn,
            pins_fn,
            nearest_node_fn,
            distance_fields,
            wet_room_names,
            weights,
        )
        if best_rot is not None:
            return float(x), float(y), best_rot, score
    return None
