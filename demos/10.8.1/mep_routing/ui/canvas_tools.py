"""Pygame renderers for canvas-local tools and their stable hit bounds."""

from __future__ import annotations

import math

import pygame

from .controls import draw_weight_view_switch
from .controls import dwelling_selector_bounds
from .drawing import draw_dashed_polyline


def terminal_tool_buttons(canvas_left, canvas_top, canvas_width):
    """Return terminal preference controls in their canvas-relative positions."""
    x = canvas_left + canvas_width - 145
    y = canvas_top + 92
    return (
        ("point", pygame.Rect(x, y, 132, 70), "Terminal"),
        ("area", pygame.Rect(x, y + 82, 132, 70), "Term. area"),
        ("map", pygame.Rect(x, y + 164, 132, 52), "Term. map"),
        ("reset", pygame.Rect(x, y + 226, 132, 34), "Reset prefs"),
    )


def draw_terminal_tool_buttons(
    screen,
    font_bold,
    font_small,
    buttons,
    active_mode,
    validity_overlay_enabled,
    *,
    text_color,
    muted_color,
    allowed_color,
):
    """Draw terminal preference controls and return their mode-to-rect mapping."""
    rects = {}
    for mode, rect, label in buttons:
        rects[mode] = rect
        active = validity_overlay_enabled if mode == "map" else active_mode == mode
        fill = (45, 54, 80) if active else (38, 44, 54)
        border = (255, 255, 255) if active else (170, 180, 190)
        pygame.draw.rect(screen, fill, rect, border_radius=10)
        pygame.draw.rect(screen, border, rect, 3 if active else 2, border_radius=10)
        label_font = font_small if mode == "reset" else font_bold
        label_surface = label_font.render(label, True, text_color)
        screen.blit(label_surface, (rect.x + 12, rect.y + (8 if mode == "reset" else 10)))
        if mode == "reset":
            continue
        if mode == "point":
            icon = pygame.Rect(rect.right - 31, rect.y + 38, 16, 16)
            pygame.draw.rect(screen, muted_color, icon, 2)
        elif mode == "area":
            icon = pygame.Rect(rect.right - 74, rect.y + 38, 58, 22)
            pygame.draw.rect(screen, (155, 89, 182), icon)
            pygame.draw.rect(screen, (255, 255, 255), icon, 2)
            for x in range(icon.left, icon.right, 8):
                pygame.draw.line(screen, (155, 89, 182), (x, icon.top), (x + 4, icon.top), 1)
                pygame.draw.line(screen, (155, 89, 182), (x, icon.bottom), (x + 4, icon.bottom), 1)
        else:
            icon = pygame.Rect(rect.right - 31, rect.y + 28, 16, 16)
            draw_dashed_polyline(
                screen,
                (icon.topleft, icon.topright, icon.bottomright, icon.bottomleft, icon.topleft),
                allowed_color,
                2,
                dash_len=4,
                gap_len=3,
            )
    return rects


def draw_canvas_tool_controls(
    screen,
    font_small,
    buttons,
    weight_switch_rect,
    *,
    ruler_enabled,
    edge_weights_enabled,
    diameter_width_enabled,
    small_weight_view,
    zoom_level,
    dwelling_label,
    dwelling_options,
    dwelling_selector_open,
    canvas_left,
    canvas_top,
    active_terminal_mode,
    text_color,
):
    """Draw common canvas controls plus their contextual status labels."""
    for action, rect, label in buttons:
        active = (
            (action == "ruler" and ruler_enabled)
            or (action == "weights" and edge_weights_enabled)
            or (action == "diameter" and diameter_width_enabled)
        )
        fill = (58, 80, 94) if active else (38, 44, 54)
        border = (52, 152, 219) if active else (170, 180, 190)
        pygame.draw.rect(screen, fill, rect, border_radius=4)
        pygame.draw.rect(screen, border, rect, 1, border_radius=4)
        label_surface = font_small.render(label, True, text_color)
        screen.blit(
            label_surface,
            (rect.centerx - label_surface.get_width() // 2, rect.centery - label_surface.get_height() // 2),
        )
    draw_weight_view_switch(screen, font_small, weight_switch_rect, small_weight_view, text_color)
    zoom_label = font_small.render(f"{zoom_level:.2f}x", True, text_color)
    screen.blit(zoom_label, (canvas_left + 12, canvas_top + 46))
    selector_bounds, option_bounds = dwelling_selector_bounds(canvas_left, canvas_top, len(dwelling_options))
    selector = pygame.Rect(selector_bounds)
    pygame.draw.rect(screen, (38, 44, 54), selector, border_radius=4)
    pygame.draw.rect(screen, (170, 180, 190), selector, 1, border_radius=4)
    label = font_small.render(f"Dwelling: {dwelling_label[:24]}", True, text_color)
    screen.blit(label, (selector.x + 8, selector.centery - label.get_height() // 2))
    arrow = font_small.render("^" if dwelling_selector_open else "v", True, text_color)
    screen.blit(arrow, (selector.right - arrow.get_width() - 8, selector.centery - arrow.get_height() // 2))
    if dwelling_selector_open:
        for option, bounds in zip(dwelling_options, option_bounds):
            rect = pygame.Rect(bounds)
            pygame.draw.rect(screen, (45, 52, 64), rect)
            pygame.draw.rect(screen, (170, 180, 190), rect, 1)
            option_label = font_small.render(option[:29], True, text_color)
            screen.blit(option_label, (rect.x + 8, rect.centery - option_label.get_height() // 2))
    if edge_weights_enabled:
        weight_view = "small pipe" if small_weight_view else "big pipe"
        weight_label = font_small.render(f"modified edge weights: {weight_view}", True, text_color)
        screen.blit(weight_label, (canvas_left + 86, canvas_top + 46))
    if active_terminal_mode:
        hint = "click add, Ctrl+click erase" if active_terminal_mode == "point" else "drag area, Ctrl+drag erase"
        terminal_label = font_small.render(f"terminal {active_terminal_mode}: {hint}", True, text_color)
        screen.blit(terminal_label, (canvas_left + 12, canvas_top + 102))


def draw_ruler_overlay(screen, font_small, start_mm, end_mm, world_to_screen, *, text_color):
    """Render a temporary world-space measurement line and its millimetre label."""
    if start_mm is None or end_mm is None:
        return
    start_px = world_to_screen(start_mm[0], start_mm[1])
    end_px = world_to_screen(end_mm[0], end_mm[1])
    length_mm = math.hypot(end_mm[0] - start_mm[0], end_mm[1] - start_mm[1])
    pygame.draw.line(screen, (255, 255, 255), start_px, end_px, 5)
    pygame.draw.line(screen, (52, 152, 219), start_px, end_px, 3)
    pygame.draw.circle(screen, (255, 255, 255), start_px, 6)
    pygame.draw.circle(screen, (52, 152, 219), start_px, 4)
    pygame.draw.circle(screen, (255, 255, 255), end_px, 6)
    pygame.draw.circle(screen, (52, 152, 219), end_px, 4)
    label = font_small.render(f"{length_mm:.0f} mm", True, text_color)
    mid_x = (start_px[0] + end_px[0]) // 2
    mid_y = (start_px[1] + end_px[1]) // 2
    pad = 4
    label_rect = pygame.Rect(mid_x + 10, mid_y - label.get_height() - 8, label.get_width() + 2 * pad, label.get_height() + 2 * pad)
    pygame.draw.rect(screen, (22, 22, 30), label_rect, border_radius=4)
    pygame.draw.rect(screen, (52, 152, 219), label_rect, 1, border_radius=4)
    screen.blit(label, (label_rect.x + pad, label_rect.y + pad))
