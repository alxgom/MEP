from types import SimpleNamespace

from mep_routing.ui.pygame_events import WorkbenchEventAdapter


def test_pending_auto_placement_updates_run_and_records_auto_marker_once():
    events = []
    app = SimpleNamespace(
        auto_placement_mode_idx=2,
        run_auto_placement=lambda: events.append("place"),
        solve_ventilation_routing=lambda: ([('Bath', [])], "Success", 7.0, 12),
        record_current_solution=lambda routes, elapsed, label, color: events.append(
            (label, elapsed)
        ),
    )
    adapter = WorkbenchEventAdapter(app, screen=None, initial_run=([], "Initial", 0.0, 0))

    adapter.run_pending_auto_placement()
    adapter.run_pending_auto_placement()

    assert adapter.state.needs_auto_placement is False
    assert adapter.state.routes == [('Bath', [])]
    assert events == ["place", ("Auto", 7.0)]


def test_machine_selector_click_only_solves_for_runtime_invalidation():
    calls = []
    app = SimpleNamespace(
        auto_placement_mode_idx=0,
        handle_machine_selector_click=lambda _pos: (True, False),
        solve_ventilation_routing=lambda: calls.append("solve") or ([], "OK", 0.0, 0),
    )
    adapter = WorkbenchEventAdapter(app, screen=None, initial_run=([], "Initial", 0.0, 0))

    adapter._left_down((25, 548))
    assert calls == []

    app.handle_machine_selector_click = lambda _pos: (True, True)
    adapter._left_down((25, 548))
    assert calls == ["solve"]
