from vent_router.ui.solution_logs import (
    best_log_updates,
    manual_best_values,
    metric_value_for_log,
    solution_kpis,
    solution_log_action,
    visible_solution_log_entries,
)


def test_solution_kpis_uses_supplied_metric_functions():
    routes = [("A", [((0, 0), (3, 4))])]

    kpis = solution_kpis(
        routes,
        elapsed_ms=12.5,
        crossings_fn=lambda _routes: 2,
        length_fn=lambda _routes: 5.0,
        turns_fn=lambda _routes: 3,
        short_pieces_fn=lambda _routes: 1,
        score_fn=lambda _routes, crossings: 100 + crossings,
    )

    assert kpis == {
        "length_m": 5.0,
        "turns": 3,
        "turns_per_m": 0.6,
        "crossings": 2,
        "short_pieces": 1,
        "score": 102,
        "elapsed_ms": 12.5,
    }


def test_solution_kpis_handles_empty_routes():
    kpis = solution_kpis(
        [],
        elapsed_ms=1,
        crossings_fn=lambda _routes: 99,
        length_fn=lambda _routes: 0.0,
        turns_fn=lambda _routes: 99,
        short_pieces_fn=lambda _routes: 99,
        score_fn=lambda _routes, _crossings: 99,
    )

    assert kpis["score"] == 0
    assert kpis["turns_per_m"] == 0.0
    assert kpis["elapsed_ms"] == 1.0


def test_metric_value_for_log_reads_known_kpi_metric():
    entry = {"kpis": {"score": 1, "length_m": 2, "turns": 3, "crossings": 4, "short_pieces": 5, "elapsed_ms": 6}}

    assert metric_value_for_log(entry, "short_pieces") == 5


def test_best_log_updates_returns_only_improved_metrics():
    previous = {
        "score": {"kpis": {"score": 10, "length_m": 0, "turns": 0, "crossings": 0, "short_pieces": 0, "elapsed_ms": 0}},
    }
    entry = {
        "kpis": {
            "score": 9,
            "length_m": 20,
            "turns": 2,
            "crossings": 1,
            "short_pieces": 0,
            "elapsed_ms": 30,
        }
    }

    updates = best_log_updates(
        entry,
        hist_idx=7,
        auto_best_logs=previous,
        metric_defs=(("score", "Best score", (1, 2, 3)), ("length_m", "Best len", (4, 5, 6))),
    )

    assert [(metric, updated["id"], label, color) for metric, updated, label, color in updates] == [
        ("score", "best:score", "Best score", (1, 2, 3)),
        ("length_m", "best:length_m", "Best len", (4, 5, 6)),
    ]
    assert updates[0][1]["hist_idx"] == 7


def test_best_log_updates_ignores_missing_hist_idx():
    assert best_log_updates({"kpis": {}}, None, {}) == []


def test_manual_best_values_and_visible_entries_preserve_metric_order():
    logs = [
        {"id": 1, "kpis": {"score": 10, "length_m": 5, "turns": 3, "crossings": 2, "short_pieces": 1, "elapsed_ms": 8}},
        {"id": 2, "kpis": {"score": 8, "length_m": 6, "turns": 4, "crossings": 1, "short_pieces": 2, "elapsed_ms": 9}},
    ]
    auto_best = {
        "turns": {"id": "best:turns", "kpis": logs[1]["kpis"]},
        "score": {"id": "best:score", "kpis": logs[1]["kpis"]},
    }

    assert manual_best_values(logs, metrics=("score", "crossings")) == {"score": 8, "crossings": 1}
    assert [entry["id"] for entry in visible_solution_log_entries(auto_best, logs, metrics=("score", "turns"), manual_limit=1)] == [
        "best:score",
        "best:turns",
        2,
    ]


def test_solution_log_action_returns_button_then_matching_row():
    class Rect:
        def __init__(self, hit):
            self.hit = hit

        def collidepoint(self, _pos):
            return self.hit

    assert solution_log_action((0, 0), Rect(True), [(Rect(True), 1)]) == "log"
    assert solution_log_action((0, 0), Rect(False), [(Rect(False), 1), (Rect(True), 2)]) == 2
    assert solution_log_action((0, 0), Rect(False), [(Rect(False), 1)]) is None
