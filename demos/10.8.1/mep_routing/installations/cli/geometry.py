"""Cli-specific machine geometry and supply connector access."""

from __future__ import annotations

import math

from shapely.geometry import Polygon, box

from mep_routing.domain.machines import DIR_REV, dir_from_axis, local_axis_to_world


CLI_SUPPLY_ACCESS_LENGTH_MM = 500


def _rotate(center, point, angle_degrees):
    radians = math.radians(angle_degrees)
    x, y = point
    return (
        round(center[0] + x * math.cos(radians) - y * math.sin(radians)),
        round(center[1] + x * math.sin(radians) + y * math.cos(radians)),
    )


def cli_machine_pins(machine, center, angle_degrees):
    """Return the Cli body corners and connector origins in plan coordinates."""
    width, depth, _height = machine.body_size_mm
    pins = {
        connector.key: _rotate(center, connector.offset_mm[:2], angle_degrees)
        for connector in machine.connectors
    }
    corners = {
        "c_tl": (-width / 2.0, depth / 2.0),
        "c_tr": (width / 2.0, depth / 2.0),
        "c_br": (width / 2.0, -depth / 2.0),
        "c_bl": (-width / 2.0, -depth / 2.0),
    }
    pins.update({name: _rotate(center, point, angle_degrees) for name, point in corners.items()})
    return pins


def cli_machine_footprint(pins):
    """Return the full machine body polygon in the routing frame."""
    return Polygon([pins[name] for name in ("c_tl", "c_tr", "c_br", "c_bl")])


def cli_air_access_spec(machine, pins, angle_degrees, connector_key, stub_length_mm=20):
    connector = machine.connector(connector_key)
    world_normal = local_axis_to_world(connector.normal_xy, angle_degrees)
    pin = pins[connector.key]
    access = (
        round(pin[0] + world_normal[0] * stub_length_mm),
        round(pin[1] + world_normal[1] * stub_length_mm),
    )
    out_direction = dir_from_axis(world_normal)
    return {
        "pin": connector.key,
        "pin_point": pin,
        "access_point": access,
        "out_dir": out_direction,
        "in_dir": DIR_REV[out_direction],
    }


def cli_air_access_specs(machine, pins, angle_degrees, connector_keys=("air_out", "air_in")):
    return tuple(cli_air_access_spec(machine, pins, angle_degrees, key) for key in connector_keys)


def cli_supply_access_specs(machine, pins, angle_degrees, stub_length_mm=CLI_SUPPLY_ACCESS_LENGTH_MM):
    return (cli_air_access_spec(machine, pins, angle_degrees, "air_out", stub_length_mm),)


def cli_supply_exit_dimensions(machine, trunk_width_mm, minimum_length_factor):
    outlet = machine.connector("air_out")
    turn_width = trunk_width_mm or outlet.rectangular_size_mm[0]
    nx, ny = outlet.normal_xy
    width, depth, _ = machine.body_size_mm
    x, y, _ = outlet.offset_mm
    inset = (abs(nx) * width + abs(ny) * depth) / 2 - nx * x - ny * y
    length = math.ceil(max(CLI_SUPPLY_ACCESS_LENGTH_MM, turn_width / 2 + inset,
                           turn_width * minimum_length_factor))
    return length, turn_width


def cli_supply_exit_footprints(machine, angle_degrees, access, turn_width):
    """Return the centered straight footprint and a mitered-turn envelope on each side."""
    x, y = access["pin_point"]
    tx, ty = access["access_point"]
    half_width = max(machine.connector("air_out").rectangular_size_mm[0], turn_width) / 2
    extension = (box(x - half_width, min(y, ty), x + half_width, max(y, ty)) if x == tx
                 else box(min(x, tx), y - half_width, max(x, tx), y + half_width))
    nx, ny = local_axis_to_world(machine.connector("air_out").normal_xy, angle_degrees)
    half_turn = turn_width / 2
    turns = {}
    for sx, sy in ((-ny, nx), (ny, -nx)):
        # Reserve one duct width along the exit and the full width across it.
        a = (tx - sx * half_turn - nx * half_turn, ty - sy * half_turn - ny * half_turn)
        b = (tx + sx * turn_width + nx * half_turn, ty + sy * turn_width + ny * half_turn)
        turns[dir_from_axis((sx, sy))] = box(min(a[0], b[0]), min(a[1], b[1]),
                                              max(a[0], b[0]), max(a[1], b[1]))
    return extension, turns


def cli_clear_supply_turns(dwelling, extension, turns, context_obstacles=()):
    """Check the straight footprint and return the locally usable 90-degree exits."""
    region = dwelling.routing_region_base
    obstacles = (*dwelling.columns, *dwelling.shafts, *context_obstacles)

    def fits(footprint):
        return region.covers(footprint) and all(footprint.disjoint(item) for item in obstacles)

    return {direction: turn for direction, turn in turns.items() if fits(turn)} if fits(extension) else {}
