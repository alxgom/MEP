"""Coc adaptation for the shared core-simple graph builder."""

from mep_routing.graphs.simple import SimpleGraphInputs


def build_coc_simple_graph_inputs(
    *, routing_region, terminals, columns, shafts, machine_pins,
    access_points, epsilon_mm, axis_clearance_mm, context_obstacles=(),
):
    allowed = routing_region.buffer(-epsilon_mm, cap_style=3, join_style=2)
    if allowed.is_empty:
        allowed = routing_region
    return SimpleGraphInputs(
        allowed_region=allowed,
        required_points=tuple(terminals.values()),
        # Exported routing_region already subtracts columns and shafts. Re-adding
        # them here can isolate a required shaft-face node; only ordered
        # installation context is additional to that normalized envelope.
        obstacles=tuple(context_obstacles),
        epsilon_mm=epsilon_mm,
        axis_clearance_mm=axis_clearance_mm,
        scaffold_spacing_mm=max(600.0, epsilon_mm * 3.0),
    )
