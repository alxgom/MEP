"""Routing-history plot panel rendering."""

from __future__ import annotations


def kpi_scope_layout(window_width, panel_width, scope_options):
    """Return compact scope-pill bounds without coupling input handling to Pygame."""
    x = window_width - panel_width + 8
    label_width = 58
    bounds = []
    for key, label, _color in scope_options:
        width = max(30, len(label) * 6 + 14)
        bounds.append((key, (x + label_width, 28, width, 20)))
        label_width += width + 4
    return tuple(bounds)


def kpi_scope_hit(pos, window_width, panel_width, scope_options):
    """Return the selected KPI scope key, if a compact scope pill was clicked."""
    px, py = pos
    for key, (x, y, width, height) in kpi_scope_layout(window_width, panel_width, scope_options):
        if x <= px < x + width and y <= py < y + height:
            return key
    return None


def draw_kpi_scope_selector(screen, font, window_width, panel_width, scope_key, scope_options, text_color, muted_color):
    import pygame
    x = window_width - panel_width + 8
    screen.blit(font.render("KPI SCOPE", True, muted_color), (x, 32))
    for key, bounds in kpi_scope_layout(window_width, panel_width, scope_options):
        _key, label, color = next(option for option in scope_options if option[0] == key)
        selected = key == scope_key
        fill = color if selected else (42, 43, 54)
        outline = color if selected else (78, 79, 92)
        foreground = (20, 21, 29) if selected else text_color
        pygame.draw.rect(screen, fill, bounds, border_radius=9)
        pygame.draw.rect(screen, outline, bounds, 1, border_radius=9)
        text = font.render(label, True, foreground)
        screen.blit(text, (bounds[0] + (bounds[2] - text.get_width()) // 2, bounds[1] + 4))


def format_current_value(metric_index, value):
    """Format a plot value with its metric-specific precision and unit."""
    if metric_index == 0:
        return f"{value:.1f} m"
    if metric_index == 1:
        return str(int(round(value)))
    if metric_index == 2:
        return str(int(round(value)))
    if metric_index == 3:
        return f"{value:.2f} turns/m"
    return f"{value:.1f} ms"


def format_minimum_value(metric_index, value):
    """Use the same metric vocabulary for the compact minimum footer."""
    if metric_index == 0:
        return f"Best {value:.2f} m"
    if metric_index in (1, 2):
        return f"Best {int(round(value))}"
    if metric_index == 3:
        return f"Best {value:.2f} turns/m"
    return f"Best {value:.1f} ms"


def draw_routing_plots(screen, font_small, font_bold, font_value, font_minimum, window_width, panel_width, buffers, sample_count, event_markers, background, text_color, muted_color, scope_key="all", scope_options=()):
    import pygame
    x, width, height, gap = window_width - panel_width + 8, panel_width - 24, 104, 8
    draw_kpi_scope_selector(
        screen, font_small, window_width, panel_width, scope_key, scope_options, text_color, muted_color,
    )
    score_title = "WORKBENCH SCORE" if scope_key == "all" else "COST SCORE"
    titles = ("DUCT LENGTH", score_title, "TURNS", "TURNS / M", "SOLVER TIME")
    colors = ((46, 204, 113), (241, 196, 15), (155, 89, 182), (26, 188, 156), (52, 152, 219))
    for index, (title, buffer, color) in enumerate(zip(titles, buffers, colors)):
        y = 54 + index * (height + gap)
        pygame.draw.rect(screen, background, (x - 4, y, width + 8, height), border_radius=6)
        pygame.draw.rect(screen, (55, 55, 70), (x - 4, y, width + 8, height), 1, border_radius=6)
        screen.blit(font_bold.render(title, True, color), (x, y + 6))
        chart_y, chart_h, count = y + 26, height - 42, len(buffer)
        if count:
            values = list(buffer)
            current, minimum = values[-1], min(values)
            pct = "0.0%" if abs(minimum) < 1e-5 else f"+{max(0.0, (current - minimum) / minimum * 100):.1f}%"
            value_text = format_current_value(index, current)
            current_label = font_value.render(f"{value_text}  {pct}", True, text_color)
            screen.blit(current_label, (x + width - current_label.get_width(), y + 6))
        if count < 2:
            screen.blit(font_small.render("Move machine to trace…", True, muted_color), (x, chart_y + chart_h // 2 - 8)); continue
        maximum = max(values)
        scale = maximum if maximum > 0 else 1.0
        sx = lambda i: x + int(i / (count - 1) * width)
        sy = lambda v: chart_y + chart_h - int(v / scale * chart_h)
        minimum, minimum_i = min(values), values.index(min(values))
        for dash_x in range(x, x + width, 6): pygame.draw.line(screen, (231, 76, 60), (dash_x, sy(minimum)), (min(dash_x + 3, x + width), sy(minimum)))
        log_markers = []
        for marker_i, label, marker_color in event_markers:
            relative = marker_i - (sample_count - count)
            if not 0 <= relative < count:
                continue
            if label.startswith("L") or label.startswith("Best"):
                log_markers.append((sx(relative), sy(values[max(0, min(count - 1, relative))]), label, marker_color))
            else:
                for dash_y in range(chart_y, chart_y + chart_h, 8): pygame.draw.line(screen, marker_color, (sx(relative), dash_y), (sx(relative), min(dash_y + 4, chart_y + chart_h)), 1)
        points = [(sx(i), sy(value)) for i, value in enumerate(values)]
        pygame.draw.lines(screen, color, False, points, 2)
        pygame.draw.circle(screen, (231, 76, 60), (sx(minimum_i), sy(minimum)), 4)
        pygame.draw.circle(screen, (255, 255, 255), points[-1], 4); pygame.draw.circle(screen, color, points[-1], 3)
        for marker_x, marker_y, _label, marker_color in log_markers:
            diamond = [(marker_x, marker_y - 5), (marker_x + 5, marker_y), (marker_x, marker_y + 5), (marker_x - 5, marker_y)]
            pygame.draw.polygon(screen, marker_color, diamond); pygame.draw.polygon(screen, (255, 255, 255), diamond, 1)
        minimum_label = font_minimum.render(format_minimum_value(index, minimum), True, (231, 76, 60))
        screen.blit(minimum_label, (x + width - minimum_label.get_width(), chart_y + chart_h + 2))
