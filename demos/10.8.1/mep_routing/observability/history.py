"""Pure session-history calculations for routing observations."""


def history_sample(routes, crossings_count, elapsed_ms, length_fn, turns_fn, score_fn):
    length_m = length_fn(routes)
    turns = turns_fn(routes) if routes else 0
    return {"length_m": length_m, "score": score_fn(routes, crossings_count) if routes else 0, "turns": turns, "turns_per_m": turns / length_m if length_m > 0 else 0.0, "elapsed_ms": float(elapsed_ms)}


def clear_history_buffers(*buffers):
    for buffer in buffers:
        buffer.clear()
