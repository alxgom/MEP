"""Pygame drawing adapters for terminal candidate validity feedback."""

from __future__ import annotations

import math


def draw_terminal_validity_square(screen, center, side, allowed, allowed_color, blocked_color, hatch_color, draw_dashed_polyline):
    """Draw one allowed or blocked terminal candidate marker."""
    import pygame

    x, y = center
    half = max(2, side // 2)
    rect = pygame.Rect(int(x - half), int(y - half), half * 2, half * 2)
    if allowed:
        draw_dashed_polyline(
            screen, [rect.topleft, rect.topright, rect.bottomright, rect.bottomleft, rect.topleft],
            allowed_color, 1, dash_len=4, gap_len=3,
        )
        return
    pygame.draw.rect(screen, blocked_color, rect, 2)
    previous_clip = screen.get_clip()
    screen.set_clip(rect)
    for offset in range(-rect.height, rect.width + rect.height, 6):
        pygame.draw.line(screen, hatch_color, (rect.left + offset, rect.bottom), (rect.left + offset + rect.height, rect.top), 1)
    screen.set_clip(previous_clip)


def draw_terminal_validity_overlay(screen, entries, world_to_screen, canvas_bounds, marker_side, allowed_color, blocked_color, hatch_color, draw_dashed_polyline):
    """Draw visible terminal validity markers in canvas coordinates."""
    canvas_left, canvas_top, canvas_width, canvas_height = canvas_bounds
    for x, y, _node_index, allowed in entries:
        screen_x, screen_y = world_to_screen(x, y)
        if screen_x < canvas_left - marker_side or screen_x > canvas_left + canvas_width + marker_side:
            continue
        if screen_y < canvas_top - marker_side or screen_y > canvas_top + canvas_height + marker_side:
            continue
        draw_terminal_validity_square(
            screen, (screen_x, screen_y), marker_side, allowed, allowed_color, blocked_color,
            hatch_color, draw_dashed_polyline,
        )


def draw_terminal_validity_tooltip(screen, font_small, mouse_position, canvas_bounds, world_from_screen, node_index_for_world, nodes, reasons_by_node, world_to_screen, window_size, text_color):
    """Draw a candidate-node validity tooltip when the cursor is close enough."""
    import pygame

    mouse_x, mouse_y = mouse_position
    canvas_left, canvas_top, canvas_width, canvas_height = canvas_bounds
    if not (canvas_left <= mouse_x <= canvas_left + canvas_width and canvas_top <= mouse_y <= canvas_top + canvas_height):
        return
    node_index = node_index_for_world(world_from_screen(mouse_x, mouse_y))
    if node_index is None or node_index < 0 or node_index >= len(nodes):
        return
    node_point = nodes[node_index]
    screen_x, screen_y = world_to_screen(float(node_point[0]), float(node_point[1]))
    if math.hypot(mouse_x - screen_x, mouse_y - screen_y) > 14:
        return
    reasons = reasons_by_node.get(node_index, ["terminal status unknown"])
    surfaces = [font_small.render(line, True, text_color) for line in [f"node {node_index}"] + reasons[:3]]
    width = max(surface.get_width() for surface in surfaces) + 18
    height = len(surfaces) * 18 + 12
    rect = pygame.Rect(mouse_x + 14, mouse_y + 14, width, height)
    window_width, window_height = window_size
    if rect.right > window_width - 8:
        rect.right = mouse_x - 14
    if rect.bottom > window_height - 8:
        rect.bottom = mouse_y - 14
    pygame.draw.rect(screen, (32, 34, 38), rect, border_radius=5)
    pygame.draw.rect(screen, (130, 138, 146), rect, 1, border_radius=5)
    for index, surface in enumerate(surfaces):
        screen.blit(surface, (rect.x + 9, rect.y + 7 + index * 18))
