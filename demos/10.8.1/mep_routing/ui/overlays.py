"""Small Pygame canvas overlays for the routing workbench."""

from __future__ import annotations


def draw_wet_room_outer_accents(screen, geometries, world_to_screen, polygon_iterator, color):
    import pygame
    for geometry in geometries:
        for polygon in polygon_iterator(geometry):
            pygame.draw.lines(screen, color, True, [world_to_screen(x, y) for x, y in polygon.exterior.coords], 3)


def draw_terminal_area_drag(screen, start_world, end_world, world_to_screen):
    import pygame
    if start_world is None or end_world is None:
        return
    x1, y1 = world_to_screen(start_world[0], start_world[1])
    x2, y2 = world_to_screen(end_world[0], end_world[1])
    rect = pygame.Rect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    if rect.width <= 1 or rect.height <= 1:
        return
    overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    overlay.fill((155, 89, 182, 55))
    screen.blit(overlay, rect.topleft)
    pygame.draw.rect(screen, (255, 255, 255), rect, 2)
