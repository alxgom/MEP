from vent_router.graphs import merge_close_values


def test_merge_close_values_deduplicates_and_thresholds_values():
    assert merge_close_values([0, 10, 15, 100], threshold=20) == [0, 100]


def test_merge_close_values_preserves_required_values():
    assert merge_close_values([0, 10, 15, 100], threshold=20, preserve_values={15}) == [0, 15, 100]


def test_merge_close_values_prefers_next_priority_value():
    assert merge_close_values([0, 30, 40, 100], threshold=20, priority_values={40}) == [0, 40, 100]


def test_merge_close_values_handles_empty_input():
    assert merge_close_values([], threshold=20) == []
