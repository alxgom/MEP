from mep_routing.installations.sal.controller import SalRoutingControllerContext, solve_routing
from mep_routing.installations.sal.orchestration import SalRoutingStrategy


def make_context(**overrides):
    values = dict(
        preflight_error=lambda: None,
        grid_available=lambda: True,
        machine_pins=lambda: {"left_mid": (0, 0)},
        refresh_graph=lambda _pins: None,
        snap_pins=lambda _pins: {"left_mid": 1},
        shaft_entry_nodes=lambda: ([3], 3),
        terminal_nodes=lambda *_args: {"Bathroom": 7},
        block_terminal_edges=lambda _weights, _nodes: {(7, 8)},
        add_shaft_clearance_weights=lambda _weights: None,
        run_shaft_search=lambda *_args: ([3, 2], 0.0, "left_mid", {"pin": "left_mid"}),
        terminals={"Bathroom": (5, 5)},
        machine_center=(0, 0),
        routing_strategy=SalRoutingStrategy.MIN_COST_FLOW_SMALL_PINS,
        bend_cost=100.0,
        ordered_room_names=lambda _terminals, _center: ["Bathroom"],
        run_small_pin_flow=lambda *_args: (True, [("Shaft", [])], "ok", 2),
        run_two_stage_flow=lambda *_args: (False, None, "unused", 0),
        run_negotiated=lambda *_args: None,
        run_sequential=lambda *_args: (False, None, "unused", 0),
        count_crossings=lambda _routes: 0,
        score_routes=lambda _routes, _crossings: 0.0,
        conflict_summary=lambda _routes: "0 crossings",
        clock=lambda: 1.0,
    )
    values.update(overrides)
    return SalRoutingControllerContext(**values)


def test_controller_stops_before_graph_work_when_preflight_blocks():
    calls = []
    result = solve_routing(make_context(preflight_error=lambda: "Blocked: Machine outside region", refresh_graph=lambda _pins: calls.append(1)))

    assert result.routes is None
    assert result.status.startswith("Blocked: Machine outside region")
    assert calls == []


def test_controller_returns_flow_success_and_overlay_edges_explicitly():
    result = solve_routing(make_context())

    assert result.routes == [("Shaft", [])]
    assert result.total_nodes == 2
    assert result.excluded_overlay_edges == frozenset({(7, 8)})
    assert result.status.startswith("Success: Min-cost flow small pins (0 crossings)")


def test_controller_keeps_lowest_scoring_sequential_candidate():
    routes_by_order = {
        ("Bathroom", "Kitchen"): [("first", [])],
        ("Kitchen", "Bathroom"): [("second", [])],
    }
    result = solve_routing(make_context(
        routing_strategy=SalRoutingStrategy.BEST_FIT,
        terminals={"Bathroom": (1, 1), "Kitchen": (2, 2)},
        ordered_room_names=lambda *_args: ["Bathroom", "Kitchen"],
        run_sequential=lambda order, *_args: (True, routes_by_order[tuple(order)], "ok", len(order)),
        count_crossings=lambda _routes: 1,
        score_routes=lambda routes, _crossings: 10 if routes[0][0] == "first" else 2,
    ))

    assert result.routes == [("second", [])]
    assert result.total_nodes == 2
    assert "tried 2 perms" in result.status
