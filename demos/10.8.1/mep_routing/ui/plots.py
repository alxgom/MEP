"""Routing-history plot panel rendering."""

from __future__ import annotations


def draw_routing_plots(screen, font_small, font_bold, window_width, panel_width, buffers, sample_count, event_markers, background, text_color, muted_color):
    import pygame
    x, width, height, gap = window_width - panel_width + 8, panel_width - 24, 104, 8
    titles = ("DUCT LENGTH  (m)", "COST SCORE", "TURNS", "TURNS / METRE", "SOLVER TIME (ms)")
    colors = ((46, 204, 113), (241, 196, 15), (155, 89, 182), (26, 188, 156), (52, 152, 219))
    for index, (title, buffer, color) in enumerate(zip(titles, buffers, colors)):
        y = 50 + index * (height + gap)
        pygame.draw.rect(screen, background, (x - 4, y, width + 8, height), border_radius=6)
        pygame.draw.rect(screen, (55, 55, 70), (x - 4, y, width + 8, height), 1, border_radius=6)
        screen.blit(font_bold.render(title, True, color), (x, y + 6))
        chart_y, chart_h, count = y + 26, height - 42, len(buffer)
        if count < 2:
            screen.blit(font_small.render("Move machine to trace…", True, muted_color), (x, chart_y + chart_h // 2 - 8)); continue
        values, maximum = list(buffer), max(buffer)
        scale = maximum if maximum > 0 else 1.0
        sx = lambda i: x + int(i / (count - 1) * width)
        sy = lambda v: chart_y + chart_h - int(v / scale * chart_h)
        minimum, minimum_i = min(values), values.index(min(values))
        for dash_x in range(x, x + width, 6): pygame.draw.line(screen, (231, 76, 60), (dash_x, sy(minimum)), (min(dash_x + 3, x + width), sy(minimum)))
        for marker_i, label, marker_color in event_markers:
            relative = marker_i - (sample_count - count)
            if 0 <= relative < count and not (label.startswith("L") or label.startswith("Best")):
                for dash_y in range(chart_y, chart_y + chart_h, 8): pygame.draw.line(screen, marker_color, (sx(relative), dash_y), (sx(relative), min(dash_y + 4, chart_y + chart_h)), 1)
        points = [(sx(i), sy(value)) for i, value in enumerate(values)]
        pygame.draw.lines(screen, color, False, points, 2)
        pygame.draw.circle(screen, (231, 76, 60), (sx(minimum_i), sy(minimum)), 4)
        pygame.draw.circle(screen, (255, 255, 255), points[-1], 4); pygame.draw.circle(screen, color, points[-1], 3)
        current = values[-1]
        pct = "0.0%" if abs(minimum) < 1e-5 else f"+{max(0.0, (current - minimum) / minimum * 100):.1f}%"
        screen.blit(font_small.render(f"Cur: {current:.1f} ({pct})", True, text_color), (x, chart_y + chart_h + 2))
        minimum_label = font_small.render(f"Min: {minimum:.1f}", True, (231, 76, 60))
        screen.blit(minimum_label, (x + width - minimum_label.get_width(), chart_y + chart_h + 2))
