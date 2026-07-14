from types import SimpleNamespace

from mep_routing.installations.sal import run_sequential_routing
from mep_routing.installations.sal import select_two_stage_routing
from mep_routing.installations.sal.route_plan import build_sal_route_plan


def test_sequential_strategy_routes_sal_large_routes_before_small_routes():
    search_calls = []

    def route_start_nodes(route_name):
        return {"Kitchen": [10], "Bathroom": [20]}[route_name]

    def run_search(_env, start_nodes, pin_names, *_args, **_kwargs):
        search_calls.append((start_nodes, list(pin_names)))
        pin_name = pin_names[0]
        return [start_nodes[0], start_nodes[0] + 1], 0.0, pin_name, {"pin": pin_name}

    def route_segments(route_name, path, *_args):
        return [(route_name, tuple(path))]

    success, routes, status, total_nodes = run_sequential_routing(
        ["Bathroom"],
        {"left_mid": [], "right_mid": [], "tl": [], "tr": [], "bl": [], "br": []},
        {},
        1,
        "left_mid",
        {"pin": "left_mid"},
        [1, 2],
        route_plan=build_sal_route_plan({"Kitchen": (1, 0), "Bathroom": (2, 0)}, (0, 0)),
        env=SimpleNamespace(adj={}),
        machine_angle=0,
        bend_cost=100,
        route_start_nodes=route_start_nodes,
        route_segments_from_path=route_segments,
        run_search=run_search,
        terminal_node_indices=lambda *_args: {},
        set_terminal_block_weight=lambda *_args: None,
        add_route_clearance_weights=lambda *_args: None,
        add_route_interaction_weights=lambda *_args: None,
        route_diameter=lambda _route_name: 90,
        route_axis_records=lambda *_args: [],
    )

    assert success is True
    assert status == "Success"
    assert [route_name for route_name, _ in routes] == ["Shaft", "Kitchen", "Bathroom"]
    assert total_nodes == 6
    assert search_calls == [([10], ["right_mid"]), ([20], ["tl", "tr", "bl", "br"])]


def test_two_stage_strategy_selects_the_lowest_scoring_complete_candidate():
    result = select_two_stage_routing(
        lambda: (True, [("Shaft", [])], "big-first", 3),
        lambda: (True, [("Kitchen", [])], "small-first", 4),
        lambda _routes: 2,
        lambda routes, crossings: len(routes) + crossings,
    )

    assert result == (True, [("Shaft", [])], "big-first", 3)
