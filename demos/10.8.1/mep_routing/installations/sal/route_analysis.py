"""Sal route-quality policy built on installation-neutral routing metrics."""

from __future__ import annotations

from dataclasses import dataclass

from mep_routing.routing import (
    RouteScoreWeights,
    append_allowed_region_warning,
    buffered_radius_mm,
    count_segment_clearance_conflicts,
    count_solution_short_pieces,
    required_clearance_mm,
    route_conflict_summary,
    route_quality_warnings,
    score_routes,
)


@dataclass(frozen=True)
class SalRouteAnalysis:
    """Evaluate Sal routes using one machine and solver-policy snapshot."""

    machine_spec: object
    policy: object
    routing_region: object = None
    shaft_extraction: object = None

    def route_diameter(self, route_name):
        return self.machine_spec.route_diameter_mm(route_name)

    def buffered_radius(self, diameter_mm):
        return buffered_radius_mm(diameter_mm, self.policy.duct_buffer_ratio)

    def required_clearance(self, diameter_a, diameter_b):
        return required_clearance_mm(diameter_a, diameter_b, self.policy.duct_buffer_ratio)

    def min_piece_length(self, route_name, terminal_segment=False):
        multiplier = 1.0 if terminal_segment else 2.0
        return self.route_diameter(route_name) * multiplier * self.policy.min_piece_factor

    def count_clearance_conflicts(self, routes):
        return count_segment_clearance_conflicts(routes, self.route_diameter, self.required_clearance)

    def count_short_pieces(self, routes):
        return count_solution_short_pieces(routes, self.min_piece_length)

    def score(self, routes, crossings=None):
        weights = RouteScoreWeights(
            bend=self.policy.bend_cost,
            crossing=self.policy.crossing_penalty,
            overlap=self.policy.overlap_score_penalty,
            clearance=self.policy.clearance_penalty,
            short_piece=self.policy.short_piece_score_penalty,
        )
        return score_routes(
            routes,
            weights,
            self.route_diameter,
            self.required_clearance,
            self.min_piece_length,
            crossings=crossings,
        )

    def quality_warnings(self, routes):
        warnings = route_quality_warnings(
            routes, self.route_diameter, self.required_clearance, self.min_piece_length,
        )
        return append_allowed_region_warning(
            warnings, routes, self.routing_region, self.shaft_extraction,
        )

    def conflict_summary(self, routes):
        return route_conflict_summary(
            routes, self.route_diameter, self.required_clearance, self.min_piece_length,
        )
