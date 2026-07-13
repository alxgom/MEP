from __future__ import annotations

DEFAULT_BEST_METRICS = (
    ("score", "Best score", (241, 196, 15)),
    ("length_m", "Best len", (46, 204, 113)),
    ("turns", "Best turns", (155, 89, 182)),
    ("crossings", "Best x", (230, 126, 34)),
    ("short_pieces", "Best short", (26, 188, 156)),
    ("elapsed_ms", "Best ms", (52, 152, 219)),
)

SOLUTION_LOG_METRICS = tuple(metric for metric, _label, _color in DEFAULT_BEST_METRICS)


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


def manual_log_entry(snapshot, log_id, hist_idx):
    entry = dict(snapshot)
    entry.update(id=log_id, hist_idx=hist_idx, kind="manual")
    return entry


def replace_history_marker(markers, label, index, color):
    return [marker for marker in markers if marker[1] != label] + [(index, label, color)]


def manual_best_values(solution_logs, metrics=SOLUTION_LOG_METRICS):
    if not solution_logs:
        return {}
    return {
        metric: min(metric_value_for_log(entry, metric) for entry in solution_logs)
        for metric in metrics
    }


def visible_solution_log_entries(auto_best_logs, solution_logs, metrics=SOLUTION_LOG_METRICS, manual_limit=3):
    auto_entries = [auto_best_logs[metric] for metric in metrics if metric in auto_best_logs]
    return auto_entries + list(solution_logs[-manual_limit:])


def solution_log_action(pos, log_button_rect, log_row_rects):
    if log_button_rect.collidepoint(pos):
        return "log"
    for rect, log_id in log_row_rects:
        if rect.collidepoint(pos):
            return log_id
    return None


def _entry_prefix(entry):
    if entry.get("kind") != "auto":
        return f"L{entry['id']}"
    return {
        "score": "Best score",
        "length_m": "Best len",
        "turns": "Best turns",
        "crossings": "Best cross",
        "short_pieces": "Best short",
        "elapsed_ms": "Best time",
    }.get(entry.get("metric"), "Best")


def draw_solution_logs_panel(
    screen,
    font_small,
    font_bold,
    *,
    window_width,
    window_height,
    panel_width,
    solution_logs,
    auto_best_logs,
    selected_log_id,
    plot_background_color,
    text_color,
    muted_color,
):
    """Draw the solution-log widget and return its clickable rectangles."""
    import pygame

    px = window_width - panel_width + 8
    y = 703
    width = panel_width - 24
    height = window_height - y - 10
    empty_rect = pygame.Rect(0, 0, 1, 1)
    if height < 90:
        return empty_rect, []

    pygame.draw.rect(screen, plot_background_color, (px - 4, y, width + 8, height), border_radius=6)
    pygame.draw.rect(screen, (55, 55, 70), (px - 4, y, width + 8, height), 1, border_radius=6)

    label = font_bold.render("SOLUTION LOGS", True, (255, 255, 255))
    screen.blit(label, (px, y + 8))
    log_button_rect = pygame.Rect(px + width - 62, y + 6, 58, 24)
    pygame.draw.rect(screen, (58, 80, 94), log_button_rect, border_radius=4)
    pygame.draw.rect(screen, (170, 180, 190), log_button_rect, 1, border_radius=4)
    button_label = font_small.render("Log", True, text_color)
    screen.blit(
        button_label,
        (log_button_rect.centerx - button_label.get_width() // 2, log_button_rect.centery - button_label.get_height() // 2),
    )

    entries = visible_solution_log_entries(auto_best_logs, solution_logs)
    if not entries:
        empty = font_small.render("No logged states", True, muted_color)
        screen.blit(empty, (px, y + 42))
        return log_button_rect, []

    best_manual = manual_best_values(solution_logs)
    row_rects = []
    row_y = y + 40
    row_height = 30
    for entry in entries:
        rect = pygame.Rect(px, row_y, width, row_height - 3)
        active = entry["id"] == selected_log_id
        fill = (45, 54, 80) if active else ((38, 42, 48) if entry.get("kind") == "auto" else (30, 34, 42))
        border = (255, 255, 255) if active else (55, 55, 70)
        pygame.draw.rect(screen, fill, rect, border_radius=4)
        pygame.draw.rect(screen, border, rect, 1, border_radius=4)
        row_rects.append((rect, entry["id"]))

        kpis = entry["kpis"]
        title = font_small.render(f"{_entry_prefix(entry)}  {int(kpis['score'])}", True, text_color)
        detail = font_small.render(
            f"{kpis['length_m']:.2f}m T{kpis['turns']} X{kpis['crossings']} S{kpis['short_pieces']}",
            True,
            muted_color,
        )
        screen.blit(title, (rect.x + 6, rect.y + 3))
        screen.blit(detail, (rect.x + 6, rect.y + 15))

        if entry.get("kind") == "manual":
            badges = [
                badge
                for metric, badge in (("score", "S"), ("length_m", "L"), ("turns", "T"), ("crossings", "X"), ("short_pieces", "P"))
                if abs(metric_value_for_log(entry, metric) - best_manual.get(metric, float("inf"))) < 1e-9
            ]
            if badges:
                badge_text = font_small.render(" ".join(badges[:3]), True, (241, 196, 15))
                screen.blit(badge_text, (rect.right - badge_text.get_width() - 6, rect.y + 3))

        row_y += row_height
        if row_y + row_height > y + height:
            break

    return log_button_rect, row_rects
