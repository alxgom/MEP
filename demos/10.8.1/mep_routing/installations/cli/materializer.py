"""Materialize Cli Steiner results into physical duct segments."""

from __future__ import annotations

from dataclasses import dataclass

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
    branch_height_mm=150, edge_sizes=None,
):
    connector = machine.connector("air_out" if role == "supply" else "air_in")
    connector_width, connector_height = connector.rectangular_size_mm
    route_name = "Supply Air Tree" if role == "supply" else "Return Air Tree"
    edge_sizes = edge_sizes or {}
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
