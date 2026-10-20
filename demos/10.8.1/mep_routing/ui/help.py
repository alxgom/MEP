"""Pygame helpers for contextual help, status, and viewer legends."""

from __future__ import annotations


HELP_TEXT = {
    "auto": ("[A] Auto-placement on/off", "[P] Cycle placement mode", "[U] Rotation mode", "[V] Placement heatmap", "[H] Heatmap scale", "[B] Heatmap palette", "[W] Placement weights"),
    "solver": ("[C] Routing strategy", "[L] Router backend", "Hard minima: no flow solvers", "Sal minimum: diameter factor", "Cli hard minimum: mm slider", "Cli minimum reset: trunk width", "[Y] A* heuristic", "[Tab] Grid type", "[G] Grid mesh", "[T] Start mode", "Terminal: click nearest node", "Term. area: drag rectangle", "Ctrl removes preferences", "Reset prefs clears all", "Sliders: piece/bend/cross", "[M] Edge weights", "[N] Small/big weight view", "[X] Real pipe width"),
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


def draw_viewer_legend(
    screen, font, canvas_bounds, show_terminal_validity, plan_label_color,
    wet_room_color, wall_color, draw_validity_square, installation_entries=(),
):
    import pygame
    canvas_left, canvas_top, _canvas_width, canvas_height = canvas_bounds
    x, bottom = canvas_left + 18, canvas_top + canvas_height - 10
    room_label = font.render("ROOMS", True, (96, 102, 110))
    wet_label = font.render("wet rooms", True, plan_label_color)
    room_rect = pygame.Rect(x, bottom - 42, max(116, wet_label.get_width() + 58), 32)
    pygame.draw.rect(screen, (248, 247, 243), room_rect, border_radius=4)
    pygame.draw.rect(screen, (140, 146, 150), room_rect, 1, border_radius=4)
    screen.blit(room_label, (room_rect.x + 8, room_rect.y + 3))
    pygame.draw.line(screen, wet_room_color, (room_rect.x + 10, room_rect.bottom - 9), (room_rect.x + 36, room_rect.bottom - 9), 3)
    pygame.draw.line(screen, wall_color, (room_rect.x + 10, room_rect.bottom - 6), (room_rect.x + 36, room_rect.bottom - 6), 1)
    screen.blit(wet_label, (room_rect.x + 44, room_rect.bottom - wet_label.get_height() - 3))

    next_bottom = room_rect.y - 8
    if installation_entries:
        labels = [(label, font.render(label + (" (active)" if active else ""), True, plan_label_color), color)
                  for label, color, active in installation_entries]
        width = max(130, max(surface.get_width() for _label, surface, _color in labels) + 40)
        install_rect = pygame.Rect(x, next_bottom - (18 + len(labels) * 18), width, 18 + len(labels) * 18)
        pygame.draw.rect(screen, (248, 247, 243), install_rect, border_radius=4)
        pygame.draw.rect(screen, (140, 146, 150), install_rect, 1, border_radius=4)
        screen.blit(font.render("INSTALLATIONS", True, (96, 102, 110)), (install_rect.x + 8, install_rect.y + 3))
        for index, (_label, surface, color) in enumerate(labels):
            row_y = install_rect.y + 18 + index * 18
            pygame.draw.line(screen, color, (install_rect.x + 10, row_y + 7), (install_rect.x + 30, row_y + 7), 4)
            screen.blit(surface, (install_rect.x + 36, row_y))
        next_bottom = install_rect.y - 8

    if show_terminal_validity:
        allowed = font.render("allowed", True, plan_label_color)
        blocked = font.render("blocked", True, plan_label_color)
        rect = pygame.Rect(x, next_bottom - 42, 34 + allowed.get_width() + 28 + blocked.get_width() + 18, 42)
        pygame.draw.rect(screen, (248, 247, 243), rect, border_radius=4)
        pygame.draw.rect(screen, (140, 146, 150), rect, 1, border_radius=4)
        screen.blit(font.render("TERMINAL VALIDITY VIEW", True, (96, 102, 110)), (rect.x + 8, rect.y + 3))
        row_y = rect.bottom - 11
        draw_validity_square(screen, (rect.x + 15, row_y), 12, True)
        screen.blit(allowed, (rect.x + 28, row_y - allowed.get_height() // 2))
        blocked_x = rect.x + 34 + allowed.get_width() + 22
        draw_validity_square(screen, (blocked_x, row_y), 12, False)
        screen.blit(blocked, (blocked_x + 14, row_y - blocked.get_height() // 2))
