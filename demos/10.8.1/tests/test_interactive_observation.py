from types import SimpleNamespace

from mep_routing.observability import InteractiveObservationController


def _app():
    terminal = SimpleNamespace(preferred_points_by_room={"Bath": [(1, 2)]}, preferred_areas=[], restored=None)
    terminal.restore_preferences = lambda points, areas: setattr(terminal, "restored", (points, areas))
    app = SimpleNamespace(
        _total_route_length_m=lambda routes: 1.0, count_solution_turns=lambda routes: 2,
        count_segment_crossings=lambda routes: 0, count_solution_short_pieces=lambda routes: 0,
        get_solution_score=lambda routes, crossings: 3,
        dwelling_source_idx=0, real_scenario_idx=1, routing_frame_idx=2,
        current_scenario_label="case", machine_cx=10, machine_cy=20, machine_angle=90,
        graph_type_idx=1, routing_strategy_idx=2, router_backend_idx=3, heuristic_mode_idx=4,
        rotation_mode_idx=1, room_start_mode_idx=0, weight_mode_idx=1, edge_weight_view_mode_idx=0,
        route_real_diameter_width_enabled=True, min_piece_factor=1.5,
        C_BEND=200.0, crossing_penalty_multiplier=5.0,
        C_BEND_DEFAULT=100.0, CROSSING_MULTIPLIER_DEFAULT=2.0, terminal_runtime=terminal,
        get_machine_pins=lambda *pose: pose, rebuilt=None,
        solve_ventilation_routing=lambda: ([], "Solved", 1.0, 4),
    )
    app.build_grid = lambda **kwargs: setattr(app, "rebuilt", kwargs)
    return app


def test_snapshot_round_trip_restores_live_fields_and_preferences():
    app = _app()
    controller = InteractiveObservationController(app, SimpleNamespace())
    snapshot = controller.runtime.snapshot(controller.current_state(), {}, "Saved", 5, 8)
    snapshot["id"] = 7
    app.machine_cx, app.C_BEND = -1, -1

    result = controller.restore_solution_log(snapshot)

    assert (app.machine_cx, app.machine_cy, app.machine_angle) == (10.0, 20.0, 90)
    assert app.C_BEND == 200.0
    assert app.terminal_runtime.restored == ({"Bath": [(1, 2)]}, [])
    assert app.rebuilt == {"machine_pins": (10.0, 20.0, 90)}
    assert controller.logs.selected_log_id == 7
    assert result[1] == "Solved"

