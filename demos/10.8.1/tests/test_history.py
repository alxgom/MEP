from mep_routing.observability import clear_history_buffers, history_sample


def test_history_sample_and_buffer_reset():
    sample = history_sample([("Bath", [])], 2, 18.5, lambda _routes: 4.0, lambda _routes: 6, lambda _routes, crossings: crossings * 10)
    first, second = [1], [2]
    clear_history_buffers(first, second)
    assert sample == {"length_m": 4.0, "score": 20, "turns": 6, "turns_per_m": 1.5, "elapsed_ms": 18.5}
    assert first == second == []
