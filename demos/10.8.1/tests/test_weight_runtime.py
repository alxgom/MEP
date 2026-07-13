import numpy as np
from shapely.geometry import box

from mep_routing.routing.weight_runtime import (
    EdgeWeightOverlay,
    RoutingWeightRuntimeContext,
    StaticClearanceCache,
    add_route_clearance_weights,
    record_weight_overlay,
    static_clearance_fields,
)


def _context(*, shafts=()):
    return RoutingWeightRuntimeContext(
        edge_list=[(0, 1, 0.0, "E"), (1, 2, 0.0, "E")],
        edge_coords=np.array([[0.0, 0.0, 10.0, 0.0], [10.0, 0.0, 20.0, 0.0]]),
        nodes=np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]),
        routing_region=box(-100, -100, 100, 100),
        room_polygons=(),
        walls=(),
        wall_polygons=(),
        shafts=shafts,
        machine_center=(1000.0, 1000.0),
        machine_angle_deg=0.0,
        machine_overall_width_mm=100.0,
        machine_body_height_mm=100.0,
        buffer_ratio=1.0,
        shaft_clearance_mm=100.0,
        machine_soft_margin_mm=20.0,
        crossing_penalty=20.0,
        clearance_penalty=10.0,
        block_weight=1000.0,
        route_diameter=lambda _name: 40.0,
    )


def test_static_fields_reuse_cache_until_static_geometry_changes(monkeypatch):
    context = _context()
    cache = StaticClearanceCache()
    calls = 0
    from mep_routing.routing import weight_runtime

    original = weight_runtime.static_clearance_distances

    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(weight_runtime, "static_clearance_distances", counted)

    first = static_clearance_fields(context, cache)
    second = static_clearance_fields(context, cache)

    assert calls == 1
    assert second[0] is first[0]

    changed = _context(shafts=(box(30, 0, 31, 1),))
    static_clearance_fields(changed, cache)
    assert calls == 2


def test_route_clearance_allows_shaft_entry_only_for_shaft_route():
    context = _context(shafts=(box(4, -2, 6, 2),))
    cache = StaticClearanceCache()

    room_weights = {}
    add_route_clearance_weights(room_weights, "Bath", context, cache)

    shaft_weights = {}
    add_route_clearance_weights(shaft_weights, "Shaft", context, cache)

    assert room_weights[(0, 1)] == context.block_weight
    assert (0, 1) not in shaft_weights


def test_overlay_omits_excluded_blocks_and_normalizes_penalty_ratio():
    context = _context()
    overlay = EdgeWeightOverlay(excluded_edges={(0, 1)})

    record_weight_overlay({(0, 1): 1000.0, (1, 2): 30.0}, context, overlay)

    assert overlay.values == {(1, 2): 2.0}
