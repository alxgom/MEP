"""Materialize Cli Steiner results into physical duct segments."""

from __future__ import annotations

from dataclasses import dataclass
import math

from mep_routing.routing import segments_from_edges

from .sizing import canonical_edge


@dataclass(frozen=True)
class CliDuctSegment:
    start: tuple[float, float]
    end: tuple[float, float]
    width_mm: int
    height_mm: int
    kind: str
    route_name: str
    sizing: object | None = None

    @property
    def line(self):
        return self.start, self.end


def materialize_cli_tree(
    *, role, machine, pins, access_spec, graph_nodes, tree, grille_connections,
    branch_height_mm=150, edge_sizes=None, machine_sizing=None,
    grille_sizes=None, grille_stub_length_mm=100,
    machine_stub_length_mm=200, machine_min_branch_length_mm=150,
):
    connector = machine.connector("air_out" if role == "supply" else "air_in")
    connector_width, connector_height = connector.rectangular_size_mm
    route_name = "Supply Air Tree" if role == "supply" else "Return Air Tree"
    edge_sizes = edge_sizes or {}
    grille_sizes = grille_sizes or {}
    segments = []
    for (left, right), (start, end) in zip(
        tree.tree_edges, segments_from_edges(graph_nodes, tree.tree_edges),
    ):
        sizing = edge_sizes.get(canonical_edge(int(left), int(right)))
        width = connector_width if sizing is None else sizing.width_mm
        height = connector_height if sizing is None else sizing.height_mm
        segments.append(CliDuctSegment(
            start, end, width, height, "tree", route_name, sizing,
        ))
    tree_lines = {segment.line for segment in segments}
    pin, access = pins[connector.key], access_spec["access_point"]
    if pin != access:
        distance = math.dist(pin, access)
        split_machine = (
            machine_sizing is not None
            and distance > machine_stub_length_mm
            and distance - machine_stub_length_mm > machine_min_branch_length_mm
        )
        if split_machine:
            split = _point_at_distance(pin, access, machine_stub_length_mm)
            segments.insert(0, CliDuctSegment(
                split, tuple(map(float, access)), machine_sizing.width_mm,
                machine_sizing.height_mm, "machine_branch", route_name, machine_sizing,
            ))
            machine_end = split
        else:
            machine_end = tuple(map(float, access))
        machine_height = (
            machine_sizing.height_mm if machine_sizing is not None else connector_height
        )
        segments.insert(0, CliDuctSegment(
            tuple(map(float, pin)), machine_end, connector_width,
            machine_height, "machine_stub", route_name,
        ))
    for connection in grille_connections:
        if connection.grille_point == connection.routing_point:
            continue
        routing_point = tuple(map(float, connection.routing_point))
        grille_point = tuple(map(float, connection.grille_point))
        if (routing_point, grille_point) in tree_lines:
            # The tree already carries the final run to a grille on its trunk.
            continue
        sizing = grille_sizes.get(connection.route_name)
        distance = math.dist(routing_point, grille_point)
        if sizing is not None and distance > grille_stub_length_mm:
            split = _point_at_distance(grille_point, routing_point, grille_stub_length_mm)
            segments.append(CliDuctSegment(
                routing_point, split, sizing.width_mm, sizing.height_mm,
                "grille_branch", connection.route_name, sizing,
            ))
            grille_start = split
        else:
            grille_start = routing_point
        segments.append(CliDuctSegment(
            grille_start, grille_point, connection.width_mm, branch_height_mm,
            "grille_stub", connection.route_name,
        ))
    return tuple(segments)


def _point_at_distance(start, end, distance):
    """Return a plan point ``distance`` millimetres from start toward end."""
    total = math.dist(start, end)
    ratio = float(distance) / total
    return (
        float(start[0]) + (float(end[0]) - float(start[0])) * ratio,
        float(start[1]) + (float(end[1]) - float(start[1])) * ratio,
    )
