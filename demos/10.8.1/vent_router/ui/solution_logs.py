from __future__ import annotations


DEFAULT_BEST_METRICS = (
    ("score", "Best score", (241, 196, 15)),
    ("length_m", "Best len", (46, 204, 113)),
    ("turns", "Best turns", (155, 89, 182)),
    ("crossings", "Best x", (230, 126, 34)),
    ("short_pieces", "Best short", (26, 188, 156)),
    ("elapsed_ms", "Best ms", (52, 152, 219)),
)


def solution_kpis(routes, elapsed_ms, crossings_fn, length_fn, turns_fn, short_pieces_fn, score_fn):
    crossings = crossings_fn(routes) if routes else 0
    length_m = length_fn(routes)
    turns = turns_fn(routes) if routes else 0
    short_pieces = short_pieces_fn(routes) if routes else 0
    return {
        "length_m": length_m,
        "turns": turns,
        "turns_per_m": turns / length_m if length_m > 0 else 0.0,
        "crossings": crossings,
        "short_pieces": short_pieces,
        "score": score_fn(routes, crossings) if routes else 0,
        "elapsed_ms": float(elapsed_ms),
    }


def metric_value_for_log(entry, metric):
    kpis = entry["kpis"]
    return {
        "score": kpis["score"],
        "length_m": kpis["length_m"],
        "turns": kpis["turns"],
        "crossings": kpis["crossings"],
        "short_pieces": kpis["short_pieces"],
        "elapsed_ms": kpis["elapsed_ms"],
    }[metric]


def best_log_updates(entry_base, hist_idx, auto_best_logs, metric_defs=DEFAULT_BEST_METRICS):
    updates = []
    if hist_idx is None:
        return updates

    for metric, label, color in metric_defs:
        current_value = metric_value_for_log(entry_base, metric)
        previous = auto_best_logs.get(metric)
        if previous is not None and current_value >= metric_value_for_log(previous, metric) - 1e-9:
            continue
        entry = dict(entry_base)
        entry["id"] = f"best:{metric}"
        entry["kind"] = "auto"
        entry["metric"] = metric
        entry["hist_idx"] = hist_idx
        updates.append((metric, entry, label, color))
    return updates
