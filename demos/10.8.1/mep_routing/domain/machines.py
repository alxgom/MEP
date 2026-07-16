from __future__ import annotations

import math
from dataclasses import dataclass, field

DIR_RIGHT, DIR_LEFT, DIR_UP, DIR_DOWN = "E", "W", "N", "S"
DIR_REV = {DIR_RIGHT: DIR_LEFT, DIR_LEFT: DIR_RIGHT, DIR_UP: DIR_DOWN, DIR_DOWN: DIR_UP}


@dataclass(frozen=True)
class MachineSpec:
    """Machine and connector dimensions in integer millimetres."""

    name: str
    body_width_mm: int
    body_height_mm: int
    overall_width_mm: int
    small_duct_diameter_mm: int
    large_duct_diameter_mm: int
    small_pin_stub_length_mm: int
    large_pin_stub_length_mm: int
    large_route_names: frozenset[str] = field(default_factory=frozenset)
    installation_height_mm: float | None = None
    installation_clearance_mm: float = 0.0

    def route_diameter_mm(self, route_name: str) -> int:
        if route_name in self.large_route_names:
            return self.large_duct_diameter_mm
        return self.small_duct_diameter_mm

    def pin_stub_length_mm(self, pin_name: str) -> int:
        if pin_name in ("left_mid", "right_mid"):
            return self.large_pin_stub_length_mm
        return self.small_pin_stub_length_mm


def _rotate_local_point(cx, cy, px, py, angle_deg):
    rad = math.radians(angle_deg)
    gx = cx + px * math.cos(rad) - py * math.sin(rad)
    gy = cy + px * math.sin(rad) + py * math.cos(rad)
    return round(gx), round(gy)


def machine_pins(spec: MachineSpec, cx, cy, angle_deg):
    w, h = spec.overall_width_mm, spec.body_height_mm
    small_y = h / 2.0 - spec.small_duct_diameter_mm / 2.0
    local_pins = {
        "left_mid": (-w / 2, 0.0),
        "right_mid": (w / 2, 0.0),
        "tl": (-w / 2, small_y),
        "tr": (w / 2, small_y),
        "bl": (-w / 2, -small_y),
        "br": (w / 2, -small_y),
    }
    global_pins = {
        name: _rotate_local_point(cx, cy, px, py, angle_deg)
        for name, (px, py) in local_pins.items()
    }

    corners = {
        "c_tl": (-w / 2, h / 2),
        "c_tr": (w / 2, h / 2),
        "c_br": (w / 2, -h / 2),
        "c_bl": (-w / 2, -h / 2),
    }
    global_corners = {
        name: _rotate_local_point(cx, cy, px, py, angle_deg)
        for name, (px, py) in corners.items()
    }
    return {**global_pins, **global_corners}


def local_axis_to_world(local_vec, machine_angle):
    lx, ly = local_vec
    rad = math.radians(machine_angle)
    gx = lx * math.cos(rad) - ly * math.sin(rad)
    gy = lx * math.sin(rad) + ly * math.cos(rad)
    if abs(gx) >= abs(gy):
        return (1.0 if gx > 0 else -1.0, 0.0)
    return (0.0, 1.0 if gy > 0 else -1.0)


def dir_from_axis(vec):
    x, y = vec
    if abs(x) >= abs(y):
        return DIR_RIGHT if x > 0 else DIR_LEFT
    return DIR_UP if y > 0 else DIR_DOWN


def port_access_specs(spec: MachineSpec, global_pins, machine_angle):
    allowed_local_dirs = {
        "left_mid": [(-1.0, 0.0)],
        "right_mid": [(1.0, 0.0)],
        "tl": [(-1.0, 0.0), (0.0, 1.0)],
        "tr": [(1.0, 0.0), (0.0, 1.0)],
        "bl": [(-1.0, 0.0), (0.0, -1.0)],
        "br": [(1.0, 0.0), (0.0, -1.0)],
    }
    specs = []
    for pin_name, local_dirs in allowed_local_dirs.items():
        if pin_name not in global_pins:
            continue
        pin_pt = global_pins[pin_name]
        stub_length = spec.pin_stub_length_mm(pin_name)
        for local_dir in local_dirs:
            wx, wy = local_axis_to_world(local_dir, machine_angle)
            access_pt = (
                round(float(pin_pt[0]) + wx * stub_length),
                round(float(pin_pt[1]) + wy * stub_length),
            )
            out_dir = dir_from_axis((wx, wy))
            specs.append({
                "pin": pin_name,
                "pin_point": (float(pin_pt[0]), float(pin_pt[1])),
                "access_point": access_pt,
                "out_dir": out_dir,
                "in_dir": DIR_REV[out_dir],
            })
    return specs


def outward_vector(pin_name, machine_angle):
    rad = math.radians(machine_angle)
    is_left = pin_name in ("left_mid", "tl", "bl")
    local_normal = (-1.0, 0.0) if is_left else (1.0, 0.0)

    gx = local_normal[0] * math.cos(rad) - local_normal[1] * math.sin(rad)
    gy = local_normal[0] * math.sin(rad) + local_normal[1] * math.cos(rad)

    if abs(gx) > abs(gy):
        return DIR_RIGHT if gx > 0 else DIR_LEFT
    return DIR_UP if gy > 0 else DIR_DOWN
