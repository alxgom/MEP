from __future__ import annotations

import math

import pygame

from vent_router.geometry import iter_polygons


def draw_geometry_overlay(screen, geometries, color_rgba, world_to_screen, surface_size):
    if not geometries:
        return
    overlay = pygame.Surface(surface_size, pygame.SRCALPHA)
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        for poly in iter_polygons(geom):
            coords = [world_to_screen(x, y) for x, y in poly.exterior.coords]
            if len(coords) >= 3:
                pygame.draw.polygon(overlay, color_rgba, coords)
    screen.blit(overlay, (0, 0))


def draw_polygon_hatch(screen, poly, color, world_to_screen, surface_size, spacing=10):
    if poly is None or poly.is_empty:
        return
    surface_width, surface_height = surface_size
    for part in iter_polygons(poly):
        coords = [world_to_screen(x, y) for x, y in part.exterior.coords]
        if len(coords) < 3:
            continue
        clip = pygame.Surface(surface_size, pygame.SRCALPHA)
        pygame.draw.polygon(clip, (255, 255, 255, 255), coords)
        hatch = pygame.Surface(surface_size, pygame.SRCALPHA)
        min_x = max(0, min(x for x, _ in coords) - 20)
        max_x = min(surface_width, max(x for x, _ in coords) + 20)
        min_y = max(0, min(y for _, y in coords) - 20)
        max_y = min(surface_height, max(y for _, y in coords) + 20)
        for x in range(min_x - (max_y - min_y), max_x + spacing, spacing):
            pygame.draw.line(hatch, color, (x, max_y), (x + (max_y - min_y), min_y), 1)
        hatch.blit(clip, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(hatch, (0, 0))


def draw_dashed_polyline(screen, points, color, width=1, dash_len=8, gap_len=5):
    if len(points) < 2:
        return
    for p1, p2 in zip(points, points[1:]):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        ux = dx / length
        uy = dy / length
        travelled = 0.0
        while travelled < length:
            seg_end = min(travelled + dash_len, length)
            start = (int(round(x1 + ux * travelled)), int(round(y1 + uy * travelled)))
            end = (int(round(x1 + ux * seg_end)), int(round(y1 + uy * seg_end)))
            pygame.draw.line(screen, color, start, end, width)
            travelled += dash_len + gap_len


def draw_outlined_text(screen, font, text, pos, color, outline_color):
    x, y = pos
    for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        screen.blit(font.render(text, True, outline_color), (x + ox, y + oy))
    screen.blit(font.render(text, True, color), (x, y))
