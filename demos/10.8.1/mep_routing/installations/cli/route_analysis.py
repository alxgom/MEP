"""Cli-owned post-route scoring for interactive comparison plots."""

from __future__ import annotations

from mep_routing.routing import count_segment_crossings, count_solution_turns, total_route_length


class CliRouteAnalysis:
    """Score Cli geometry with the same bend/crossing costs exposed by its solver."""

    def __init__(self, bend_penalty_mm, crossing_penalty_multiplier):
        self.bend_penalty_mm = float(bend_penalty_mm)
        self.crossing_penalty_multiplier = float(crossing_penalty_multiplier)

    def score(self, routes, crossings=None):
        crossings = count_segment_crossings(routes) if crossings is None else int(crossings)
        turns = count_solution_turns(routes)
        crossing_penalty = self.bend_penalty_mm * self.crossing_penalty_multiplier
        return (
            float(total_route_length(routes))
            + self.bend_penalty_mm * turns
            + crossing_penalty * crossings
        )
