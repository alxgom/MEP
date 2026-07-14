import pytest

from mep_routing.routing.search import (
    SearchBackend,
    coerce_search_backend,
    run_super_sink_search,
)


def test_search_backend_keeps_the_existing_selector_indices():
    assert coerce_search_backend(0) is SearchBackend.STATE_ASTAR
    assert coerce_search_backend(1) is SearchBackend.LINE_GRAPH_ASTAR
    assert coerce_search_backend(2) is SearchBackend.LINE_GRAPH_GBFS
    with pytest.raises(ValueError):
        coerce_search_backend(3)


@pytest.mark.parametrize(
    ("backend", "expected_greedy"),
    [
        (SearchBackend.LINE_GRAPH_ASTAR, False),
        (SearchBackend.LINE_GRAPH_GBFS, True),
    ],
)
def test_line_graph_backend_selection_only_changes_greedy_mode(monkeypatch, backend, expected_greedy):
    calls = []

    def fake_line_search(*args, **kwargs):
        calls.append((args, kwargs))
        return "line-result"

    monkeypatch.setattr("mep_routing.routing.search.run_super_sink_line_graph_search", fake_line_search)

    result = run_super_sink_search(
        backend,
        "env",
        [1],
        ["pin"],
        {"pin": []},
        50.0,
        edge_weights={(1, 2): 3.0},
        heuristic_mode=2,
        machine_center=(4.0, 5.0),
        estimate_turns_fn="turns",
    )

    assert result == "line-result"
    args, kwargs = calls[0]
    assert args == ("env", [1], ["pin"], {"pin": []}, 50.0)
    assert kwargs == {
        "edge_weights": {(1, 2): 3.0},
        "heuristic_mode": 2,
        "machine_center": (4.0, 5.0),
        "estimate_turns_fn": "turns",
        "greedy": expected_greedy,
    }


def test_state_backend_preserves_search_arguments(monkeypatch):
    calls = []

    def fake_state_search(*args, **kwargs):
        calls.append((args, kwargs))
        return "state-result"

    monkeypatch.setattr("mep_routing.routing.search.run_super_sink_state_astar", fake_state_search)

    result = run_super_sink_search(
        SearchBackend.STATE_ASTAR,
        "env",
        1,
        ["pin"],
        {"pin": []},
        25.0,
    )

    assert result == "state-result"
    args, kwargs = calls[0]
    assert args == ("env", 1, ["pin"], {"pin": []}, 25.0)
    assert kwargs["edge_weights"] is None
    assert kwargs["heuristic_mode"] == 0
    assert kwargs["machine_center"] == (0.0, 0.0)
