"""Materialize Cli Steiner results into physical duct segments."""

from __future__ import annotations

from dataclasses import dataclass

from mep_routing.routing import segments_from_edges


@dataclass(frozen=True)
class CliDuctSegment:
    start: tuple[float, float]
    end: tuple[float, float]
    width_mm: int
    height_mm: int
    kind: str
    route_name: str

    @property
    def line(self):
        return self.start, self.end


def materialize_cli_tree(
    *, role, machine, pins, access_spec, graph_nodes, tree, grille_connections,
    branch_height_mm=150,
):
    connector = machine.connector("air_out" if role == "supply" else "air_in")
    connector_width, connector_height = connector.rectangular_size_mm
    route_name = "Supply Air Tree" if role == "supply" else "Return Air Tree"
    segments = [
        CliDuctSegment(start, end, connector_width, connector_height, "tree", route_name)
        for start, end in segments_from_edges(graph_nodes, tree.tree_edges)
    ]
    pin, access = pins[connector.key], access_spec["access_point"]
    if pin != access:
        segments.insert(0, CliDuctSegment(
            tuple(map(float, pin)), tuple(map(float, access)), connector_width,
            connector_height, "machine_stub", route_name,
        ))
    for connection in grille_connections:
        if connection.grille_point == connection.routing_point:
            continue
        segments.append(CliDuctSegment(
            tuple(map(float, connection.routing_point)), tuple(map(float, connection.grille_point)),
            connection.width_mm, branch_height_mm, "grille_branch", connection.route_name,
        ))
    return tuple(segments)
