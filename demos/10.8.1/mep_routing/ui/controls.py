from __future__ import annotations

import math


def canvas_tool_button_bounds(canvas_left, canvas_top):
    """Return stable bounds for shared canvas toolbar controls."""
    size = 28
    gap = 6
    x0 = canvas_left + 12
    y0 = canvas_top + 12
    reset_width = 46
    cursor = x0 + 2 * (size + gap) + reset_width + gap
    ruler = (cursor, y0, 58, size)
    weights = (ruler[0] + ruler[2] + gap, y0, 72, size)
    weight_switch = weight_view_switch_bounds(weights)
    diameter = (weight_switch[0] + weight_switch[2] + gap, y0, 54, size)
    return (
        ("in", (x0, y0, size, size), "+"),
        ("out", (x0 + size + gap, y0, size, size), "-"),
        ("reset", (x0 + 2 * (size + gap), y0, reset_width, size), "1:1"),
        ("ruler", ruler, "Ruler"),
        ("weights", weights, "Weights"),
        ("diameter", diameter, "Diam"),
    )


def weight_view_switch_bounds(weights_bounds, gap=6):
    x, y, width, _height = weights_bounds
    return x + width + gap, y + 2, 92, 24


def slider_value_from_x(x, rect, min_value, max_value):
    """Map an x-coordinate to a clamped slider value."""
    if rect.width <= 0:
        return min_value
    t = (float(x) - rect.x) / max(1.0, float(rect.width))
    t = max(0.0, min(1.0, t))
    return min_value + t * (max_value - min_value)


def slider_fraction(value, min_value, max_value):
    span = max_value - min_value
    if span <= 0:
        return 0.0
    return max(0.0, min(1.0, (value - min_value) / span))


def draw_min_piece_slider(screen, font_small, x, y, width, value, min_value, max_value, text_color, muted_color):
    """Draw the min-piece slider and return its Pygame hit rectangle."""
    import pygame

    rect = pygame.Rect(int(x), int(y + 18), int(width), 8)
    label = font_small.render(f"Min pieces factor: {value:.2f}x", True, text_color)
    screen.blit(label, (x, y))
    pygame.draw.rect(screen, (22, 22, 30), rect, border_radius=4)
    pygame.draw.rect(screen, muted_color, rect, 1, border_radius=4)
    knob_x = int(rect.x + slider_fraction(value, min_value, max_value) * rect.width)
    pygame.draw.circle(screen, (255, 255, 255), (knob_x, rect.centery), 8)
    pygame.draw.circle(screen, (190, 196, 204), (knob_x, rect.centery), 8, 1)
    min_label = font_small.render(f"{min_value:.1f}", True, muted_color)
    max_label = font_small.render(f"{max_value:.1f}", True, muted_color)
    screen.blit(min_label, (rect.x, rect.bottom + 3))
    screen.blit(max_label, (rect.right - max_label.get_width(), rect.bottom + 3))
    return rect


def draw_weight_slider(
    screen,
    font_small,
    x,
    y,
    width,
    label,
    value,
    min_value,
    max_value,
    text_color,
    muted_color,
    suffix="",
    integer=False,
):
    """Draw a weight slider and its reset button, returning both hit rectangles."""
    import pygame

    reset_size = 18
    slider_width = int(width) - reset_size - 8
    rect = pygame.Rect(int(x), int(y + 18), int(slider_width), 8)
    reset_rect = pygame.Rect(rect.right + 8, int(y + 13), reset_size, reset_size)
    value_text = f"{int(round(value))}" if integer else f"{value:.1f}"
    label_surface = font_small.render(f"{label}: {value_text}{suffix}", True, text_color)
    screen.blit(label_surface, (x, y))
    pygame.draw.rect(screen, (22, 22, 30), rect, border_radius=4)
    pygame.draw.rect(screen, muted_color, rect, 1, border_radius=4)
    knob_x = int(rect.x + slider_fraction(value, min_value, max_value) * rect.width)
    pygame.draw.circle(screen, (255, 255, 255), (knob_x, rect.centery), 8)
    pygame.draw.circle(screen, (190, 196, 204), (knob_x, rect.centery), 8, 1)
    pygame.draw.rect(screen, (32, 34, 38), reset_rect, border_radius=4)
    pygame.draw.rect(screen, (128, 136, 144), reset_rect, 1, border_radius=4)
    cx, cy = reset_rect.center
    icon_color = (198, 204, 210)
    pygame.draw.arc(screen, icon_color, pygame.Rect(cx - 5, cy - 5, 10, 10), math.radians(35), math.radians(315), 2)
    pygame.draw.polygon(screen, icon_color, [(cx + 4, cy - 6), (cx + 8, cy - 5), (cx + 5, cy - 2)])
    return rect, reset_rect


def draw_weight_view_switch(screen, font_small, rect, left_active, text_color):
    """Draw the shared small/big pipe edge-weight switch."""
    import pygame

    pygame.draw.rect(screen, (32, 34, 38), rect, border_radius=rect.height // 2)
    pygame.draw.rect(screen, (150, 158, 166), rect, 1, border_radius=rect.height // 2)
    knob_radius = 4 if left_active else 8
    knob_x = rect.left + 13 if left_active else rect.right - 13
    pygame.draw.circle(screen, (198, 204, 210), (knob_x, rect.centery), knob_radius)
    pygame.draw.circle(screen, (255, 255, 255), (knob_x, rect.centery), knob_radius, 1)
    label = "Small" if left_active else "Big"
    label_surface = font_small.render(label, True, text_color)
    label_x = rect.x + 28 if left_active else rect.x + 14
    screen.blit(label_surface, (label_x, rect.centery - label_surface.get_height() // 2))
