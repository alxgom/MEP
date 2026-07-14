from types import SimpleNamespace

from mep_routing.installations.sal.negotiated import (
    SalNegotiatedContext,
    run_negotiated_congestion,
)
from mep_routing.installations.sal.route_plan import build_sal_route_plan


def test_negotiated_routing_builds_sal_routes_and_stops_on_zero_crossings():
    calls = []

    def route_start_nodes(route_name):
        return {"Kitchen": [1], "Bathroom": [2]}[route_name]

    def run_search(_env, start_nodes, target_pins, *_args, **_kwargs):
        calls.append((start_nodes, tuple(target_pins)))
        pin = target_pins[0]
        return [start_nodes[0], 9], 0.0, pin, {"pin": pin}

    context = SalNegotiatedContext(
        env=SimpleNamespace(adj={}),
        route_start_nodes=route_start_nodes,
        terminal_node_indices=lambda *_args: {},
        set_terminal_block_weight=lambda *_args: None,
        add_route_clearance_weights=lambda *_args: None,
        add_route_interaction_weights=lambda *_args: None,
        route_diameter=lambda _route_name: 90,
        route_segments_from_path=lambda route_name, path, *_args: [(route_name, tuple(path))],
        route_axis_records=lambda *_args: [],
        run_search=run_search,
        count_crossings=lambda _routes: 0,
        score_routes=lambda routes, _crossings: len(routes),
    )

    result = run_negotiated_congestion(
        ["Bathroom"],
        {"left_mid": [], "right_mid": [], "tl": [], "tr": [], "bl": [], "br": []},
        {},
        [0],
        0,
        route_plan=build_sal_route_plan({"Kitchen": (1, 0), "Bathroom": (2, 0)}, (0, 0)),
        context=context,
        machine_angle=0,
        bend_cost=100,
    )

    assert result.success is True
    assert result.crossing_free is True
    assert result.attempts == 1
    assert result.total_nodes == 6
    assert [route_name for route_name, _ in result.routes] == ["Shaft", "Kitchen", "Bathroom"]
    assert calls == [([0], ("left_mid", "right_mid")), ([1], ("right_mid",)), ([2], ("tl", "tr", "bl", "br"))]
