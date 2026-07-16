"""Materialize Sal solver paths into route geometry for the live graph."""

from __future__ import annotations

from dataclasses import dataclass

from mep_routing.routing import (
    build_routes_from_paths,
    route_segments_from_path,
    source_start_nodes,
)


@dataclass(frozen=True)
class SalRouteMaterializer:
    env_nodes: object
    spatial_index: object
    route_plan: object
    add_shaft_entry_segments: object = None

    def source_start_nodes(self, source_spec):
        return source_start_nodes(source_spec, self.spatial_index)

    def route_segments(self, route_name, path, pin_name=None, global_pins=None, target=None):
        return route_segments_from_path(
            route_name,
            path,
            self.env_nodes,
            self.add_shaft_entry_segments,
            pin_name,
            global_pins,
            target,
            shaft_route_name=self.route_plan.shaft_route,
        )

    def build_routes(self, route_order, paths, targets, global_pins):
        return build_routes_from_paths(
            route_order,
            paths,
            targets,
            global_pins,
            self.route_segments,
        )
