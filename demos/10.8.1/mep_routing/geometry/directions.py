"""Hashable travel headings for cardinal and experimental arbitrary-angle grids."""

import math

Direction = str | tuple[float, float]

CARDINAL_VECTORS = {"E": (1.0, 0.0), "W": (-1.0, 0.0), "N": (0.0, 1.0), "S": (0.0, -1.0)}
REVERSE_DIRECTION = {"E": "W", "W": "E", "N": "S", "S": "N"}


def travel_direction(start, end):
    dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
    length = math.hypot(dx, dy)
    if length < 1e-7:
        return None
    if abs(dy) < 1e-7:
        return "E" if dx > 0 else "W"
    if abs(dx) < 1e-7:
        return "N" if dy > 0 else "S"
    return (round(dx / length, 6), round(dy / length, 6))


def reverse_direction(direction):
    if direction is None:
        return None
    if isinstance(direction, str):
        return REVERSE_DIRECTION[direction]
    return (-direction[0], -direction[1])


def same_direction(left, right):
    if left == right:
        return True
    if left is None or right is None:
        return False
    a = CARDINAL_VECTORS[left] if isinstance(left, str) else left
    b = CARDINAL_VECTORS[right] if isinstance(right, str) else right
    # Same normalized-vector tolerance as routing-core's flow_turn policy.
    return abs(a[0] - b[0]) <= 1e-4 and abs(a[1] - b[1]) <= 1e-4
