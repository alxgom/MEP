from mep_routing.ui.events import (
    CanvasGestureState,
    CanvasHit,
    PanelHit,
    PanelInteractionState,
    begin_canvas_gesture,
    begin_panel_interaction,
    end_canvas_gesture,
    end_panel_interaction,
    move_canvas_gesture,
    move_panel_interaction,
    routing_key_transition,
)


def _state(**overrides):
    state = {
        "machine_angle": 270,
        "auto_placement_mode_idx": 0,
        "routing_strategy_idx": 1,
        "room_start_mode_idx": 0,
        "router_backend_idx": 0,
        "heuristic_mode_idx": 1,
        "graph_type_idx": 2,
        "rotation_mode_idx": 0,
        "weight_mode_idx": 0,
    }
    state.update(overrides)
    return state


COUNTS = {
    "routing_strategy": 3,
    "room_start": 2,
    "router_backend": 2,
    "heuristic": 3,
    "graph": 3,
    "rotation_mode": 2,
    "auto_placement": 3,
    "weight_mode": 2,
}


def test_routing_key_transitions_cycle_solver_state_and_request_required_adapter_work():
    graph = routing_key_transition("cycle_graph", _state(), COUNTS)
    assert graph.state["graph_type_idx"] == 0
    assert graph.rebuild_graph is True
    assert graph.markers == (("Grid:0", (155, 89, 182)),)

    rotation = routing_key_transition("cycle_rotation_mode", _state(), COUNTS)
    assert rotation.state["rotation_mode_idx"] == 1
    assert rotation.apply_rotation_mode is True

    room_start = routing_key_transition("cycle_room_start", _state(room_start_mode_idx=1), COUNTS)
    assert room_start.state["room_start_mode_idx"] == 0
    assert [marker[0] for marker in room_start.markers] == ["Start:0", "Strat:1"]


def test_routing_key_transitions_preserve_auto_placement_policy_and_ignore_unknown_commands():
    rotation = routing_key_transition("rotate_machine", _state(auto_placement_mode_idx=2), COUNTS)
    assert rotation.state["machine_angle"] == 0
    assert rotation.state["auto_placement_mode_idx"] == 0

    weight = routing_key_transition("cycle_weight_mode", _state(auto_placement_mode_idx=2), COUNTS)
    assert weight.state["weight_mode_idx"] == 1
    assert weight.needs_auto_placement is True
    assert weight.refresh_placement_fields is False

    auto = routing_key_transition("toggle_auto_placement", _state(auto_placement_mode_idx=0), COUNTS)
    assert auto.state["auto_placement_mode_idx"] == 2
    assert auto.needs_auto_placement is True
    assert auto.record_history is False
    assert routing_key_transition("unhandled", _state(), COUNTS) is None


def test_canvas_gesture_begin_prioritizes_pan_ruler_terminal_and_machine_actions():
    point = (100.0, 200.0)
    screen = (10, 20)
    pan = begin_canvas_gesture(
        CanvasGestureState(ruler_mode=True, terminal_tool_mode="point"),
        world_point=point,
        screen_point=screen,
        shift=True,
        ctrl=False,
        hit=CanvasHit(machine_hit=True),
    )
    assert pan.state.panning is True
    assert pan.commands[0].name == "start_pan"

    ruler = begin_canvas_gesture(
        CanvasGestureState(ruler_mode=True, terminal_tool_mode="point"),
        world_point=point,
        screen_point=screen,
        shift=False,
        ctrl=False,
        hit=CanvasHit(machine_hit=True),
    )
    assert ruler.state.ruler_dragging is True
    assert ruler.commands[0].name == "start_ruler"

    terminal = begin_canvas_gesture(
        CanvasGestureState(terminal_tool_mode="point"),
        world_point=point,
        screen_point=screen,
        shift=False,
        ctrl=True,
        hit=CanvasHit(machine_hit=True),
    )
    assert terminal.commands[0].name == "apply_terminal_point"
    assert terminal.commands[0].value == (point, True)

    machine = begin_canvas_gesture(
        CanvasGestureState(),
        world_point=point,
        screen_point=screen,
        shift=False,
        ctrl=False,
        hit=CanvasHit(machine_hit=True),
        machine_center_mm=(90.0, 180.0),
    )
    assert machine.state.machine_drag_offset_mm == (10.0, 20.0)


def test_canvas_gesture_selects_hits_and_commits_area_on_release():
    selected = begin_canvas_gesture(
        CanvasGestureState(),
        world_point=(1.0, 1.0),
        screen_point=(1, 1),
        shift=False,
        ctrl=False,
        hit=CanvasHit(route_name="Kitchen", room_route_name="Bathroom"),
    )
    assert selected.commands == (selected.commands[0],)
    assert selected.commands[0].name == "select_route"
    assert selected.commands[0].value == "Kitchen"

    started = begin_canvas_gesture(
        CanvasGestureState(terminal_tool_mode="area"),
        world_point=(10.0, 20.0),
        screen_point=(10, 20),
        shift=False,
        ctrl=True,
        hit=CanvasHit(),
    )
    moved = move_canvas_gesture(started.state, world_point=(30.0, 40.0), screen_point=(30, 40))
    finished = end_canvas_gesture(moved.state, button="left")
    assert finished.commands[0].name == "apply_terminal_area"
    assert finished.commands[0].value == ((10.0, 20.0), (30.0, 40.0), True)
    assert finished.state.terminal_area_dragging is False
    assert finished.state.machine_dragging is False


def test_panel_interaction_uses_click_priority_before_slider_or_tool_actions():
    transition = begin_panel_interaction(
        PanelInteractionState(),
        hit=PanelHit(
            help_card="routing",
            min_piece_slider=True,
            canvas_tool="ruler",
            solution_log_action="log",
        ),
        screen_x=120,
    )
    assert transition.state.active_slider is None
    assert transition.commands[0].name == "toggle_help"
    assert transition.commands[0].value == "routing"

    reset = begin_panel_interaction(
        PanelInteractionState(),
        hit=PanelHit(bend_reset=True, canvas_tool="weights"),
        screen_x=120,
    )
    assert reset.commands[0].name == "reset_slider"
    assert reset.commands[0].value == "bend"


def test_panel_slider_drag_emits_initial_and_motion_updates_then_releases():
    started = begin_panel_interaction(
        PanelInteractionState(),
        hit=PanelHit(crossing_slider=True),
        screen_x=40,
    )
    assert started.state.active_slider == "crossing"
    assert started.commands[0].value == ("crossing", 40)

    moved = move_panel_interaction(started.state, screen_x=75)
    assert moved.commands[0].name == "set_slider"
    assert moved.commands[0].value == ("crossing", 75)

    released = end_panel_interaction(moved.state)
    assert released.state.active_slider is None
    assert released.commands == ()
