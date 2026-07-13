"""Pygame helpers for contextual help, status, and viewer legends."""

from __future__ import annotations


HELP_TEXT = {
    "auto": ("[A] Auto-placement on/off", "[P] Cycle placement mode", "[U] Rotation mode", "[V] Placement heatmap", "[H] Heatmap scale", "[B] Heatmap palette", "[W] Placement weights"),
    "solver": ("[C] Routing strategy", "[L] Router backend", "[Y] A* heuristic", "[Tab] Grid type", "[G] Grid mesh", "[T] Start mode", "Terminal: click nearest node", "Term. area: drag rectangle", "Ctrl removes preferences", "Reset prefs clears all", "Sliders: piece/bend/cross", "[M] Edge weights", "[N] Small/big weight view", "[X] Real pipe width"),
    "machine": ("Drag machine with mouse", "Wheel: rotate machine", "[U] Torque/field rotation", "Shift+wheel: zoom", "Shift+drag or middle drag: pan", "[D] Dwelling source", "[O] Routing frame"),
    "kpi": ("Plots compare current values", "against best observed minimum.", "Solver time excludes rendering."),
    "status": ("Routing status and solver time.", "Right log panel stores", "session-local states.", "[Esc] Clear selection / ruler", "[Space] New apartment"),
}


def help_lines(card_id):
    return HELP_TEXT.get(card_id)


def draw_card_help_button(screen, card_id, card_rect, font, active, muted_color, text_color):
    import pygame
    button = pygame.Rect(card_rect.right - 26, card_rect.y + 8, 18, 18)
    pygame.draw.rect(screen, (58, 80, 94) if active else (50, 55, 66), button, border_radius=9)
    pygame.draw.rect(screen, muted_color, button, 1, border_radius=9)
    label = font.render("?", True, text_color)
    screen.blit(label, (button.centerx - label.get_width() // 2, button.centery - label.get_height() // 2))
    return button


def draw_help_popup(screen, font, lines, origin, text_color):
    import pygame
    if not lines:
        return
    x, y = origin
    rect = pygame.Rect(x, y, 235, 18 + len(lines) * 18)
    pygame.draw.rect(screen, (22, 22, 30), rect, border_radius=6)
    pygame.draw.rect(screen, (120, 130, 145), rect, 1, border_radius=6)
    for index, line in enumerate(lines):
        label = font.render(line, True, text_color)
        screen.blit(label, (rect.x + 10, rect.y + 10 + index * 18))


def draw_transient_message(screen, font, message, canvas_origin, text_color):
    import pygame
    if not message:
        return
    label = font.render(message, True, text_color)
    x, y = canvas_origin
    rect = pygame.Rect(x + 16, y + 16, label.get_width() + 20, label.get_height() + 12)
    pygame.draw.rect(screen, (55, 45, 35), rect, border_radius=5)
    pygame.draw.rect(screen, (241, 196, 15), rect, 1, border_radius=5)
    screen.blit(label, (rect.left + 10, rect.top + 6))


def draw_viewer_legend(screen, font, canvas_bounds, show_terminal_validity, plan_label_color, wet_room_color, wall_color, draw_validity_square):
    import pygame
    canvas_left, canvas_top, _canvas_width, canvas_height = canvas_bounds
    x, y = canvas_left + 18, canvas_top + canvas_height - 34
    if show_terminal_validity:
        allowed = font.render("allowed", True, plan_label_color)
        blocked = font.render("blocked", True, plan_label_color)
        rect = pygame.Rect(x, y - 30, 34 + allowed.get_width() + 28 + blocked.get_width() + 18, 24)
        pygame.draw.rect(screen, (248, 247, 243), rect, border_radius=4)
        pygame.draw.rect(screen, (140, 146, 150), rect, 1, border_radius=4)
        draw_validity_square(screen, (rect.x + 15, rect.centery), 12, True)
        screen.blit(allowed, (rect.x + 28, rect.centery - allowed.get_height() // 2))
        blocked_x = rect.x + 34 + allowed.get_width() + 22
        draw_validity_square(screen, (blocked_x, rect.centery), 12, False)
        screen.blit(blocked, (blocked_x + 14, rect.centery - blocked.get_height() // 2))
    label = font.render("wet rooms", True, plan_label_color)
    rect = pygame.Rect(x, y, label.get_width() + 58, 24)
    pygame.draw.rect(screen, (248, 247, 243), rect, border_radius=4)
    pygame.draw.rect(screen, (140, 146, 150), rect, 1, border_radius=4)
    pygame.draw.line(screen, wet_room_color, (rect.x + 10, rect.centery), (rect.x + 36, rect.centery), 3)
    pygame.draw.line(screen, wall_color, (rect.x + 10, rect.centery + 3), (rect.x + 36, rect.centery + 3), 1)
    screen.blit(label, (rect.x + 44, rect.centery - label.get_height() // 2))
