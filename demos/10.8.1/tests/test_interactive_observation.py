from types import SimpleNamespace

from mep_routing.app import (
    InstallationMachineUi, InstallationUiSession, WorkbenchState, build_demo_config,
)
from mep_routing.installations.cli import CLI_INSTALLATION
from mep_routing.installations.sal import SAL_INSTALLATION
from mep_routing.observability import InteractiveObservationController
from mep_routing.ui.installation_header import InstallationHeaderState


def _app():
    terminal = SimpleNamespace(preferred_points_by_room={"Bath": [(1, 2)]}, preferred_areas=[], restored=None)
    terminal.restore_preferences = lambda points, areas: setattr(terminal, "restored", (points, areas))
    workbench_state = WorkbenchState.create()
    workbench_state.installation("sal").pose.set((10, 20), 90)
    workbench_state.installation("cli").pose.set((30, 40), 180)
    workbench_state.dwelling.terminal_runtime = terminal
    app = SimpleNamespace(
        workbench_state=workbench_state,
        demo_config=build_demo_config(SAL_INSTALLATION),
        _total_route_length_m=lambda routes: 1.0, count_solution_turns=lambda routes: 2,
        count_segment_crossings=lambda routes: 0, count_solution_short_pieces=lambda routes: 0,
        get_solution_score=lambda routes, crossings: 3,
        dwelling_source_idx=0, real_scenario_idx=1, routing_frame_idx=2,
        current_scenario_label="case",
        graph_type_idx=1, routing_strategy_idx=2, router_backend_idx=3, heuristic_mode_idx=4,
        rotation_mode_idx=1, room_start_mode_idx=0, weight_mode_idx=1, edge_weight_view_mode_idx=0,
        route_real_diameter_width_enabled=True, min_piece_factor=1.5,
        C_BEND=200.0, crossing_penalty_multiplier=5.0,
        C_BEND_DEFAULT=100.0, CROSSING_MULTIPLIER_DEFAULT=2.0,
        rebuilt=None,
        installation_header_state=InstallationHeaderState(
            order=("sal", "cli"), enabled=frozenset({"sal", "cli"}), active="cli",
        ),
        installation_ui_session=InstallationUiSession.create({
            "sal": InstallationMachineUi(SAL_INSTALLATION.machine_catalog, True),
            "cli": InstallationMachineUi(CLI_INSTALLATION.machine_catalog, True),
        }),
    )
    app.routing_controller = SimpleNamespace(
        solution_score=app.get_solution_score,
        installation_solution_score=lambda key, routes, crossings: {
            "sal": 10.0, "cli": 20.0,
        }.get(key, app.get_solution_score(routes, crossings)),
        count_short_pieces=app.count_solution_short_pieces,
        machine_pins=lambda *pose: pose,
        build_grid=lambda **kwargs: setattr(app, "rebuilt", kwargs),
        solve_installations=lambda: ([], "Solved", 1.0, 4),
    )
    return app


def test_snapshot_round_trip_restores_live_fields_and_preferences():
    app = _app()
    app.workbench_state.cli.duct_geometry_mode_idx = 2
    controller = InteractiveObservationController(app, SimpleNamespace())
    snapshot = controller.runtime.snapshot(controller.current_state(), {}, "Saved", 5, 8)
    snapshot["id"] = 7
    saved_cli_key = snapshot["installation_configs"]["cli"]["machine_key"]
    app.workbench_state.installation("sal").pose.x_mm, app.C_BEND = -1, -1
    app.workbench_state.installation("cli").pose.set((-30, -40), 0)
    app.workbench_state.cli.bend_penalty_mm = -1
    app.workbench_state.cli.duct_geometry_mode_idx = 0
    app.installation_header_state = InstallationHeaderState(
        order=("sal", "cli"), enabled=frozenset({"sal"}), active="sal",
    )

    result = controller.restore_solution_log(snapshot)

    assert app.workbench_state.installation("sal").pose.as_tuple() == (10.0, 20.0, 90)
    assert app.C_BEND == 200.0
    assert app.workbench_state.installation("cli").pose.as_tuple() == (30.0, 40.0, 180.0)
    assert app.workbench_state.cli.bend_penalty_mm == 4000.0
    assert app.workbench_state.cli.duct_geometry_mode_idx == 2
    assert app.installation_header_state.active == "cli"
    assert app.installation_header_state.enabled == frozenset({"sal", "cli"})
    assert app.installation_ui_session.selected_keys["cli"] == saved_cli_key
    assert app.workbench_state.dwelling.terminal_runtime.restored == ({"Bath": [(1, 2)]}, [])
    assert app.rebuilt == {"machine_pins": (10.0, 20.0, 90)}
    assert controller.logs.selected_log_id == 7
    assert result[1] == "Solved"


def test_kpi_histories_keep_aggregate_and_installation_samples_from_one_execution():
    app = _app()
    app._total_route_length_m = lambda routes: float(len(routes))
    app.workbench_state.replace_runs({
        "sal": SimpleNamespace(routes=(("Sal", ()),), elapsed_ms=4.0),
        "cli": SimpleNamespace(routes=(("Cli", ()),), elapsed_ms=7.0),
    })
    controller = InteractiveObservationController(app, SimpleNamespace())

    controller.record_history((("Sal", ()),), 0, 4.0)

    assert controller.kpi_scope == "all"
    assert controller.history.length_m[-1] == 2.0
    assert controller.history.elapsed_ms[-1] == 11.0
    assert controller.history.score[-1] == 30.0
    assert controller._histories["sal"].length_m[-1] == 1.0
    assert controller._histories["cli"].length_m[-1] == 1.0
    assert controller._histories["sal"].elapsed_ms[-1] == 4.0
    assert controller._histories["cli"].elapsed_ms[-1] == 7.0
    assert controller._histories["sal"].score[-1] == 10.0
    assert controller._histories["cli"].score[-1] == 20.0
    assert controller.select_kpi_scope("sal")
    assert controller.displayed_history is controller._histories["sal"]
