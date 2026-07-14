import numpy as np
from shapely.geometry import box

from mep_routing.routing.weight_runtime import (
    EdgeWeightOverlay,
    RoutingRuntime,
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

    core_weights = {}
    add_route_clearance_weights(
        core_weights, "Core", context, cache, shaft_route_name="Core",
    )
    assert (0, 1) not in core_weights


def test_overlay_omits_excluded_blocks_and_normalizes_penalty_ratio():
    context = _context()
    overlay = EdgeWeightOverlay(excluded_edges={(0, 1)})

    record_weight_overlay({(0, 1): 1000.0, (1, 2): 30.0}, context, overlay)

    assert overlay.values == {(1, 2): 2.0}


def _runtime(monkeypatch, *, backend=0):
    context = _context()
    overlay = EdgeWeightOverlay()
    env = type("Env", (), {"nodes": context.nodes, "adj": {}})()
    return RoutingRuntime(
        env,
        context,
        StaticClearanceCache(),
        overlay,
        search_backend=backend,
        heuristic_mode=2,
        bend_cost=17.0,
        estimate_turns_fn=lambda *_args: 0,
    )


def test_runtime_reuses_clearance_cache(monkeypatch):
    runtime = _runtime(monkeypatch)
    calls = 0
    from mep_routing.routing import weight_runtime

    original = weight_runtime.static_clearance_distances

    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(weight_runtime, "static_clearance_distances", counted)
    runtime.add_static_clearance_weights({}, 40.0)
    runtime.add_static_clearance_weights({}, 40.0)

    assert calls == 1


def test_runtime_terminal_blocks_are_excluded_from_overlay(monkeypatch):
    runtime = _runtime(monkeypatch)
    weights = {}

    edge = runtime.set_terminal_block_weight(weights, 1, 0)
    runtime.record_edge_weight_overlay(weights)

    assert edge == (0, 1)
    assert weights[(0, 1)] == runtime.context.block_weight
    assert runtime.overlay.values == {}


def test_runtime_dispatches_search_with_explicit_policy(monkeypatch):
    runtime = _runtime(monkeypatch, backend=2)
    captured = {}
    from mep_routing.routing import weight_runtime

    def fake_search(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return [0, 1], 10.0, "pin", {"node_idx": 1}

    monkeypatch.setattr(weight_runtime, "run_super_sink_search", fake_search)
    result = runtime.run_super_sink_search(
        [0], ["pin"], {"pin": [{"node_idx": 1}]}, edge_weights={(0, 1): 12.0},
    )

    assert result[0] == [0, 1]
    assert captured["args"][:2] == (2, runtime.env)
    assert captured["args"][5] == 17.0
    assert captured["kwargs"]["heuristic_mode"] == 2
    assert captured["kwargs"]["machine_center"] == runtime.context.machine_center
