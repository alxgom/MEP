from types import SimpleNamespace

from mep_routing.app import WorkbenchState, build_demo_config
from mep_routing.installations.sal import SAL_INSTALLATION
from mep_routing.ui.commands import WorkbenchCommandController


def _pygame_keys():
    return SimpleNamespace(K_r=1, K_c=2, K_t=3, K_l=4, K_y=5, K_TAB=6, K_a=7, K_p=8, K_u=9, K_w=10)


def _app():
    rebuilds = []
    workbench_state = WorkbenchState.create()
    workbench_state.installation("sal").pose.set((10, 20))
    app = SimpleNamespace(
        workbench_state=workbench_state,
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
