from types import SimpleNamespace

from mep_routing.app import WorkbenchState, build_demo_config
from mep_routing.installations.sal import SAL_INSTALLATION
from mep_routing.installations.cli import CLI_INSTALLATION
from mep_routing.ui.commands import WorkbenchCommandController


def _pygame_keys():
    return SimpleNamespace(K_r=1, K_c=2, K_t=3, K_l=4, K_y=5, K_TAB=6, K_a=7, K_p=8, K_u=9, K_w=10)


def _app():
    rebuilds = []
    workbench_state = WorkbenchState.create()
    workbench_state.installation("sal").pose.set((10, 20))
    app = SimpleNamespace(
        workbench_state=workbench_state,
        installation_header_state=SimpleNamespace(active="sal", enabled={"sal"}),
        CLI_INSTALLATION=CLI_INSTALLATION,
        demo_config=build_demo_config(SAL_INSTALLATION),
        auto_placement_mode_idx=0, routing_strategy_idx=0, room_start_mode_idx=0,
        router_backend_idx=0, heuristic_mode_idx=0, graph_type_idx=0,
        rotation_mode_idx=0, weight_mode_idx=0,
        ROUTING_STRATEGIES=("a", "b"), ROOM_START_MODES=("a", "b"),
        ROUTER_BACKENDS=("a", "b"), HEURISTIC_MODES=("a", "b"),
        GRAPH_TYPES=("a", "b"), ROTATION_MODES=("a", "b"),
        AUTO_PLACEMENT_MODES=("a", "b", "c"),
        shaft_extraction=None,
        get_machine_pins=lambda *args: args,
        build_grid=lambda **kwargs: rebuilds.append(kwargs),
        apply_rotation_mode_once=lambda: None,
    )
    return app, rebuilds


def test_rotate_command_updates_angle_and_preserves_transition_effects():
    app, rebuilds = _app()
    controller = WorkbenchCommandController(app, _pygame_keys())

    transition = controller.apply_routing_key_command(1)

    assert transition is not None
    assert app.workbench_state.installation("sal").pose.angle_degrees == 90
    assert rebuilds == []


def test_unknown_key_does_not_change_state():
    app, rebuilds = _app()
    controller = WorkbenchCommandController(app, _pygame_keys())
    assert controller.apply_routing_key_command(99) is None
    assert app.workbench_state.installation("sal").pose.angle_degrees == 0
    assert rebuilds == []


def test_cli_graph_hotkey_cycles_cli_state_without_changing_sal():
    app, _rebuilds = _app()
    app.installation_header_state = SimpleNamespace(active="cli", enabled={"cli"})
    app.workbench_state.cli.graph_type_idx = 3
    controller = WorkbenchCommandController(app, _pygame_keys())

    transition = controller.apply_routing_key_command(_pygame_keys().K_TAB)

    assert transition is not None
    assert app.workbench_state.cli.graph_type_idx == 0
    assert app.graph_type_idx == 0


def test_cli_router_hotkey_selects_wall_candidates_for_hard_steiner_modes():
    app, _ = _app()
    app.installation_header_state = SimpleNamespace(active="cli", enabled={"cli"})
    app.workbench_state.cli.router_backend_idx = 4
    controller = WorkbenchCommandController(app, _pygame_keys())
    for backend in (5, 6, 7):
        controller.apply_routing_key_command(_pygame_keys().K_l)
        assert app.workbench_state.cli.router_backend_idx == backend
        assert app.workbench_state.cli.room_start_mode_idx == 1
    controller.apply_routing_key_command(_pygame_keys().K_l)
    assert app.workbench_state.cli.router_backend_idx == 0
    assert app.workbench_state.cli.room_start_mode_idx == 0
    assert app.router_backend_idx == 0


def test_cli_auto_placement_hotkeys_update_only_cli_mode():
    app, _rebuilds = _app()
    app.installation_header_state = SimpleNamespace(active="cli", enabled={"cli"})
    controller = WorkbenchCommandController(app, _pygame_keys())

    manual = controller.apply_routing_key_command(_pygame_keys().K_p)
    automatic = controller.apply_routing_key_command(_pygame_keys().K_a)

    assert manual.needs_auto_placement is False
    assert automatic.needs_auto_placement is True
    assert app.workbench_state.cli.auto_placement_mode_idx == 1
    assert app.auto_placement_mode_idx == 0


def test_cli_weight_sliders_update_cli_values_without_changing_sal():
    app, _rebuilds = _app()
    app.installation_header_state = SimpleNamespace(active="cli", enabled={"cli"})
    app.canvas_ui = SimpleNamespace(slider_value=lambda name, *_args: 2300 if name == "bend" else 7.5)
    app.C_BEND = 4000
    app.crossing_penalty_multiplier = 5
    controller = WorkbenchCommandController(app, _pygame_keys())

    controller.set_bend_weight_from_slider_x(100)
    controller.set_crossing_weight_from_slider_x(100)

    assert app.workbench_state.cli.bend_penalty_mm == 2300
    assert app.workbench_state.cli.crossing_penalty_multiplier == 7.5
    assert (app.C_BEND, app.crossing_penalty_multiplier) == (4000, 5)


def test_cli_duct_geometry_selector_cycles_off_core_aligned():
    app, _rebuilds = _app()
    app.installation_header_state = SimpleNamespace(active="cli", enabled={"cli"})
    rect = SimpleNamespace(collidepoint=lambda _pos: True)
    app.canvas_ui = SimpleNamespace(
        toolbar_buttons=lambda: (("duct_geometry", rect, None),),
        apply_view_action=lambda _action: None,
    )
    controller = WorkbenchCommandController(app, _pygame_keys())

    selected = []
    modes = []
    for _ in range(3):
        selected.append(controller.handle_canvas_tool_button_click((0, 0)))
        modes.append(app.workbench_state.cli.duct_geometry_mode_idx)

    assert selected == ["duct_geometry"] * 3
    assert modes == [1, 2, 0]
