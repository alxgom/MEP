from dataclasses import dataclass, field

from mep_routing.app.interaction_runtime import (
    InteractionCallbacks,
    InteractionState,
    RoutingRun,
    RoutingTransition,
    apply_auto_placement,
    apply_dwelling_change,
    apply_rotation,
    apply_routing_transition,
    apply_slider_change,
    apply_terminal_area,
    apply_terminal_point,
    reset_terminal_preferences,
)


@dataclass
class Recorder:
    run: RoutingRun = field(default_factory=lambda: RoutingRun([("Bath", [])], "Success", 12.0, 5))
    events: list[tuple] = field(default_factory=list)

    def callbacks(self):
        return InteractionCallbacks(
            solve=lambda: self.run,
            record_solution=lambda run, label, color: self.events.append(("solution", label, color, run)),
            record_history=lambda run: self.events.append(("history", run)),
            add_marker=lambda label, color: self.events.append(("marker", label, color)),
            apply_slider=lambda name, x: (f"{name}:{x:.1f}", (1, 2, 3)),
            apply_terminal_point=lambda point, remove: (True, "Bath"),
            apply_terminal_area=lambda start, end, remove: (True, "Kitchen"),
            clear_terminal_preferences=lambda: self.events.append(("clear_preferences",)),
            rotate_machine=lambda delta: 90 if delta > 0 else 270,
            generate_dwelling=lambda: self.events.append(("generate",)),
            clear_solution_logs=lambda: self.events.append(("clear_logs",)),
            clear_history=lambda: self.events.append(("clear_history",)),
            run_auto_placement=lambda: self.events.append(("auto_place",)),
        )


def _state():
    return InteractionState(RoutingRun([], "Initial", 0.0, 0))


def test_slider_change_reroutes_and_records_solution_only_after_success():
    recorder = Recorder()

    updated = apply_slider_change(_state(), "bend", 42.0, recorder.callbacks())

    assert updated.run is recorder.run
    assert recorder.events == [("solution", "bend:42.0", (1, 2, 3), recorder.run)]


def test_blocked_reroute_does_not_record_history_marker_or_solution():
    recorder = Recorder(run=RoutingRun(None, "Blocked: no path", 5.0, 0))

    updated = apply_rotation(_state(), 90, recorder.callbacks())

    assert updated.run.status.startswith("Blocked")
    assert recorder.events == []


def test_terminal_commands_preserve_selection_and_use_distinct_solution_markers():
    recorder = Recorder()
    callbacks = recorder.callbacks()

    point_state = apply_terminal_point(_state(), (1.0, 2.0), remove=False, callbacks=callbacks)
    area_state = apply_terminal_area(point_state, (1.0, 1.0), (2.0, 2.0), remove=True, callbacks=callbacks)

    assert area_state.selected_route_name == "Kitchen"
    assert [event[1] for event in recorder.events if event[0] == "solution"] == ["Term+", "Area-"]


def test_terminal_reset_and_auto_placement_run_expected_effect_sequence():
    recorder = Recorder()
    callbacks = recorder.callbacks()

    reset_state = reset_terminal_preferences(_state(), callbacks)
    auto_state = apply_auto_placement(InteractionState(reset_state.run, needs_auto_placement=True), callbacks)

    assert auto_state.needs_auto_placement is False
    assert [event[0] for event in recorder.events] == [
        "clear_preferences", "solution", "auto_place", "history", "marker",
    ]


def test_routing_transition_propagates_auto_request_and_records_all_markers():
    recorder = Recorder()
    transition = RoutingTransition(
        needs_auto_placement=True,
        record_history=True,
        markers=(("Grid", (4, 5, 6)), ("Mode", (7, 8, 9))),
    )

    updated = apply_routing_transition(_state(), transition, recorder.callbacks())

    assert updated.needs_auto_placement is True
    assert [event[0] for event in recorder.events] == ["history", "marker", "marker"]


def test_dwelling_change_resets_sessions_then_reroutes_with_marker():
    recorder = Recorder()

    updated = apply_dwelling_change(
        _state(), recorder.callbacks(), marker=("Src:1", (52, 152, 219)), needs_auto_placement=True,
    )

    assert updated.needs_auto_placement is True
    assert [event[0] for event in recorder.events] == [
        "generate", "clear_logs", "clear_history", "history", "marker",
    ]
