"""Sal adaptation for installation-neutral graph builders."""

from shapely.geometry import Polygon

from mep_routing.graphs.simple import SimpleGraphInputs


def build_sal_simple_graph_inputs(
    *, routing_region, terminals, columns, shafts, machine_pins,
    access_points, epsilon_mm, axis_clearance_mm,
):
    """Describe Sal's required nodes and fixed obstacles for the simple grid."""
    allowed = routing_region.buffer(-epsilon_mm, cap_style=3, join_style=2)
    if allowed.is_empty:
        allowed = routing_region
    obstacles = list(columns) + list(shafts)
    corners = [machine_pins.get(name) for name in ("c_tl", "c_tr", "c_br", "c_bl")]
    if all(point is not None for point in corners):
        obstacles.append(Polygon(corners))
    return SimpleGraphInputs(
        allowed_region=allowed,
        required_points=tuple(terminals.values()) + tuple(access_points),
        obstacles=tuple(obstacles),
        epsilon_mm=epsilon_mm,
        axis_clearance_mm=axis_clearance_mm,
    )
