from mep_routing.observability import RoutingObservationRuntime


def _runtime():
    return RoutingObservationRuntime(
        route_length_m=lambda routes: float(len(routes)),
        route_turns=lambda routes: len(routes) + 1,
        route_score=lambda routes, crossings: len(routes) * 10 + crossings,
        solution_kpis=lambda routes, elapsed: {
            "length_m": float(len(routes)), "turns": len(routes) + 1,
            "turns_per_m": 2.0, "crossings": 2, "short_pieces": 0,
            "score": len(routes) * 10 + 2, "elapsed_ms": elapsed,
        },
        metric_value=lambda entry, metric: entry["kpis"][metric],
        best_metrics=(("score", "Best score", (1, 2, 3)),),
    )


def test_manual_log_records_history_auto_best_and_marker():
    runtime = _runtime()

    state = {
        "machine": (1, 2, 0),
        "preferred_terminal_points_by_room": {},
        "preferred_terminal_areas": [],
    }
    assert runtime.log_manual(state, {"Shaft": [(0, 0)]}, "Solved", 12, 40)

    assert runtime.history.sample_count == 1
    assert runtime.logs.manual_logs[0]["status"] == "Solved"
    assert runtime.logs.auto_best_logs["score"]["status"] == "Auto best"
    assert {marker[1] for marker in runtime.history.event_markers} == {"Best score", "L1"}


def test_empty_routes_are_not_logged():
    runtime = _runtime()
    assert not runtime.log_manual({}, {}, "Blocked", 4, 0)
    assert runtime.history.sample_count == 0
    assert runtime.logs.manual_logs == []
