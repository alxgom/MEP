"""Cli-specific machine geometry and supply connector access."""

from __future__ import annotations

import math

from mep_routing.domain.machines import DIR_REV, dir_from_axis, local_axis_to_world


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


def cli_supply_access_specs(machine, pins, angle_degrees, stub_length_mm=500):
    return (cli_air_access_spec(machine, pins, angle_degrees, "air_out", stub_length_mm),)
