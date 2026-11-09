"""Pygame adapters for placement, distance and edge-weight overlays."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


_PLACEMENT_VALID = (22, 172, 114)
_PLACEMENT_WALL = (236, 184, 43)
_PLACEMENT_INVALID = (207, 72, 66)


@lru_cache(maxsize=8)
def _placement_surface(mask):
    import pygame

    height, width = mask.valid.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[mask.visible] = (*_PLACEMENT_INVALID, 105)
    rgba[mask.valid] = (*_PLACEMENT_WALL, 155)
    rgba[mask.clear_room] = (*_PLACEMENT_VALID, 155)
    return pygame.image.frombytes(rgba[::-1].tobytes(), (width, height), "RGBA")


def draw_placement_mask(screen, mask, viewport):
    """Draw nearest-neighbor mask tiles, scaling only the visible portion."""
    import pygame

    surface = _placement_surface(mask)
    nx, ny = surface.get_size()
    x, y = mask.origin_mm
    left, top = viewport.to_screen(x, y + ny * mask.step_mm)
    right, bottom = viewport.to_screen(x + nx * mask.step_mm, y)
    target = pygame.Rect(left, top, right - left, bottom - top)
    previous_clip = screen.get_clip()
    clip = previous_clip.clip(viewport.canvas_rect)
    visible = target.clip(clip)
    if not visible.width or not visible.height:
        return
    a = (visible.left - left) * nx // target.width
    b = (visible.top - top) * ny // target.height
    c = math.ceil((visible.right - left) * nx / target.width)
    d = math.ceil((visible.bottom - top) * ny / target.height)
    x0, x1 = (left + round(i * target.width / nx) for i in (a, c))
    y0, y1 = (top + round(j * target.height / ny) for j in (b, d))
    cropped = surface.subsurface((a, b, c - a, d - b))
    screen.set_clip(clip)
    screen.blit(pygame.transform.scale(cropped, (x1 - x0, y1 - y0)), (x0, y0))
    screen.set_clip(previous_clip)


def draw_placement_cell_grid(screen, grid, mask, viewport):
    """Draw the fixed Cell Skeleton, with placement candidates at cell centers."""
    import pygame

    previous_clip = screen.get_clip()
    screen.set_clip(previous_clip.clip(viewport.canvas_rect))
    centers = []
    for left, bottom, right, top in grid.diagnostics["cells"]:
        x0, y0 = viewport.to_screen(left, top)
        x1, y1 = viewport.to_screen(right, bottom)
        pygame.draw.rect(screen, (153, 159, 174), (x0, y0, x1 - x0, y1 - y0), 1)
        center = ((left + right) / 2, (bottom + top) / 2)
        centers.append((viewport.to_screen(*center), mask.center_status(center)))
    points = [viewport.to_screen(*point) for point in grid.nodes]
    for u, v, *_ in grid.edges:
        pygame.draw.line(screen, (58, 75, 96), points[u], points[v])
    for point in points:
        pygame.draw.circle(screen, (58, 75, 96), point, 2)
    for point, status in centers:
        if status == "blocked":
            pygame.draw.circle(screen, (81, 91, 107), point, 3)
        else:
            color = _PLACEMENT_VALID if status == "clear" else _PLACEMENT_WALL
            pygame.draw.circle(screen, (37, 46, 60), point, 7)
            pygame.draw.circle(screen, (248, 249, 252), point, 6)
            pygame.draw.circle(screen, color, point, 4)
    screen.set_clip(previous_clip)


def draw_placement_mask_legend(screen, mask, viewport, font, angle, cells_visible=False):
    import pygame

    x, y, width, _ = horizontal_colorbar_bounds(viewport.canvas_rect, width=340)
    if cells_visible:
        y -= 19
    pygame.draw.rect(screen, (31, 35, 43), (x, y, width, 65 if cells_visible else 46), border_radius=4)
    title = font.render(f"CLI body + supply extension · {angle:g}° · {mask.step_mm} mm raster", True, (240, 240, 240))
    screen.blit(title, (x + 8, y + 4))
    items = ((8, _PLACEMENT_VALID, "Body clear"),
             (98, _PLACEMENT_WALL, "Body at wall / edge"),
             (235, _PLACEMENT_INVALID, "Not cleared"))
    for offset, color, label in items:
        pygame.draw.rect(screen, color, (x + offset, y + 26, 10, 10))
        screen.blit(font.render(label, True, (240, 240, 240)), (x + offset + 15, y + 22))
    if cells_visible:
        label = font.render("Cell Skeleton · ring = candidate center · [Shift+G]", True, (240, 240, 240))
        screen.blit(label, (x + 8, y + 42))


def horizontal_colorbar_bounds(canvas_bounds, width=240):
    """Return a compact bottom-right overlay inside the usable canvas."""
    canvas_left, canvas_top, canvas_width, canvas_height = canvas_bounds
    width = min(int(width), max(140, canvas_width - 36))
    return canvas_left + canvas_width - width - 18, canvas_top + canvas_height - 54, width, 40


def draw_distance_colorbar(screen, has_scores, canvas_bounds, heatmap_scale_mode, palette_index, heatmap_color, text_color):
    """Draw the active node-score palette legend."""
    import pygame

    if not has_scores:
        return
    panel_x, panel_y, panel_width, panel_height = horizontal_colorbar_bounds(canvas_bounds)
    pygame.draw.rect(screen, (31, 35, 43), (panel_x, panel_y, panel_width, panel_height), border_radius=4)
    pygame.draw.rect(screen, (150, 158, 166), (panel_x, panel_y, panel_width, panel_height), 1, border_radius=4)
    bar_x, bar_y, bar_width, bar_height = panel_x + 48, panel_y + 19, panel_width - 96, 10
    for x in range(bar_width):
        t = x / max(1, bar_width - 1)
        color = heatmap_color(min(1.0, t / 0.75) if heatmap_scale_mode == 0 else t)
        pygame.draw.line(screen, color, (bar_x + x, bar_y), (bar_x + x, bar_y + bar_height - 1))
    pygame.draw.rect(screen, (255, 255, 255), (bar_x - 1, bar_y - 1, bar_width + 2, bar_height + 2), 1)
    font = pygame.font.SysFont("Outfit", 14, bold=True)
    high_label = font.render("HIGH", True, heatmap_color(1.0))
    low_label = font.render("LOW", True, heatmap_color(0.0))
    screen.blit(low_label, (panel_x + 8, bar_y - 2))
    screen.blit(high_label, (panel_x + panel_width - high_label.get_width() - 8, bar_y - 2))
    title = font.render("VIRIDIS" if palette_index == 1 else "TURBO", True, text_color)
    screen.blit(title, (panel_x + panel_width // 2 - title.get_width() // 2, panel_y + 2))


def build_distance_heatmap_surface(node_scores, regular_nodes, grid_spacing_mm, canvas_bounds, screen_to_world, interpolate_score, score_to_t, heatmap_color):
    """Rasterize a semi-transparent distance field for the current canvas."""
    import pygame

    canvas_left, canvas_top, canvas_width, canvas_height = canvas_bounds
    min_score, max_score = min(node_scores.values()), max(node_scores.values())
    score_grid = {}
    for node_index, score in node_scores.items():
        if node_index >= len(regular_nodes):
            continue
        x, y = regular_nodes[node_index]
        score_grid[(round(float(x) / grid_spacing_mm), round(float(y) / grid_spacing_mm))] = float(score)
    low_width, low_height = 320, max(1, round(320 * canvas_height / canvas_width))
    low_surface = pygame.Surface((low_width, low_height), pygame.SRCALPHA)
    for pixel_y in range(low_height):
        screen_y = canvas_top + (pixel_y + 0.5) * canvas_height / low_height
        for pixel_x in range(low_width):
            screen_x = canvas_left + (pixel_x + 0.5) * canvas_width / low_width
            world_x, world_y = screen_to_world(screen_x, screen_y)
            score = interpolate_score(world_x, world_y, score_grid)
            if score is None:
                continue
            color = heatmap_color(score_to_t(score, min_score, max_score))
            low_surface.set_at((pixel_x, pixel_y), (color[0], color[1], color[2], 150))
    return pygame.transform.smoothscale(low_surface, (canvas_width, canvas_height))


def draw_edge_weight_heatmap(screen, edge_weights, nodes, adjacency, world_to_screen, block_weight, blocked_color, cool_colormap, log_scale):
    """Draw edge-weight ratios over the active graph."""
    import pygame

    if not edge_weights or nodes is None:
        return
    _max_ratio, log_max = log_scale(edge_weights, block_weight)
    for (u, v), ratio in edge_weights.items():
        if u not in adjacency or u >= len(nodes) or v >= len(nodes):
            continue
        point_a = world_to_screen(nodes[u][0], nodes[u][1])
        point_b = world_to_screen(nodes[v][0], nodes[v][1])
        if ratio >= block_weight:
            color, width = blocked_color, 5
        else:
            color, width = cool_colormap(math.log1p(max(0.0, ratio)) / log_max), 3
        pygame.draw.line(screen, color, point_a, point_b, width)


def draw_edge_weight_colorbar(screen, edge_weights, canvas_bounds, block_weight, blocked_color, cool_colormap, log_scale, text_color):
    """Draw the edge-weight ratio legend and blocked-edge key."""
    import pygame

    if not edge_weights:
        return
    panel_x, panel_y, panel_width, panel_height = horizontal_colorbar_bounds(canvas_bounds, 270)
    bar_x, bar_y, bar_width, bar_height = panel_x + 54, panel_y + 19, panel_width - 108, 10
    max_ratio, _log_max = log_scale(edge_weights, block_weight)
    pygame.draw.rect(screen, (31, 35, 43), (panel_x, panel_y, panel_width, panel_height), border_radius=4)
    pygame.draw.rect(screen, (150, 158, 166), (panel_x, panel_y, panel_width, panel_height), 1, border_radius=4)
    for x in range(bar_width):
        color = cool_colormap(x / max(1, bar_width - 1))
        pygame.draw.line(screen, color, (bar_x + x, bar_y), (bar_x + x, bar_y + bar_height - 1))
    pygame.draw.rect(screen, (255, 255, 255), (bar_x - 1, bar_y - 1, bar_width + 2, bar_height + 2), 1)
    font = pygame.font.SysFont("Outfit", 13, bold=True)
    title = font.render("EDGE WEIGHT", True, text_color)
    high_label = font.render(f"+{max_ratio:.1f}x", True, cool_colormap(1.0))
    low_label = font.render("+0x", True, cool_colormap(0.0))
    screen.blit(title, (panel_x + panel_width // 2 - title.get_width() // 2, panel_y + 2))
    screen.blit(low_label, (panel_x + 8, bar_y - 2))
    screen.blit(high_label, (panel_x + panel_width - high_label.get_width() - 8, bar_y - 2))
    if any(value >= block_weight for value in edge_weights.values()):
        block_rect = pygame.Rect(panel_x + panel_width - 42, panel_y + 4, 12, 8)
        pygame.draw.rect(screen, blocked_color, block_rect)
        blocked_label = font.render("BLOCK", True, text_color)
        screen.blit(blocked_label, (block_rect.x - blocked_label.get_width() - 4, panel_y + 2))
