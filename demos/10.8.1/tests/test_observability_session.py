from mep_routing.observability.session import RoutingHistory, SolutionLogSession


def test_routing_history_keeps_absolute_indices_after_bounded_buffer_rollover():
    history = RoutingHistory(maxlen=2)
    sample = {"length_m": 1.0, "score": 2.0, "turns": 3, "turns_per_m": 3.0, "elapsed_ms": 4.0}

    assert [history.append(sample) for _ in range(3)] == [0, 1, 2]
    assert list(history.length_m) == [1.0, 1.0]
    assert history.latest_index == 2
    assert history.add_marker("Rotate", (1, 2, 3)) is True
    assert history.event_markers == [(2, "Rotate", (1, 2, 3))]

    history.clear()
    assert history.sample_count == 0
    assert list(history.length_m) == []
    assert history.event_markers == []


def test_solution_log_session_updates_only_improved_auto_bests_and_selects_manual_entry():
    session = SolutionLogSession()
    definitions = (("score", "Best score", (1, 2, 3)),)
    metric_value = lambda entry, metric: entry["kpis"][metric]

    first = {"kpis": {"score": 10}}
    assert session.add_manual(first, 4)["id"] == 1
    assert session.selected_log_id == 1
    assert [item[0] for item in session.update_auto_bests(first, 4, definitions, metric_value)] == ["score"]
    assert session.update_auto_bests({"kpis": {"score": 10}}, 5, definitions, metric_value) == []
    updates = session.update_auto_bests({"kpis": {"score": 9}}, 6, definitions, metric_value)
    assert updates[0][1]["hist_idx"] == 6
    assert session.auto_best_logs["score"]["kpis"]["score"] == 9
