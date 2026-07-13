from mep_routing.placement.runtime import run_core_like_placement, run_topological_placement


def test_core_like_runtime_returns_selected_position_and_candidate_count():
    outcome = run_core_like_placement([], lambda _room: [], (0,), lambda *_args: True, lambda *_args: 0, lambda *_args: ((10, 20, 90, 3.5), 4))
    assert outcome.position == (10, 20)
    assert outcome.rotation == 90
    assert outcome.candidate_count == 4


def test_topological_runtime_preserves_empty_score_fields_without_selection():
    outcome = run_topological_placement(None, [], (), None, None, None, [], None, lambda *_args: ({}, {"field": []}), lambda *_args: None)
    assert outcome.position is None
    assert outcome.fields == {"field": []}
