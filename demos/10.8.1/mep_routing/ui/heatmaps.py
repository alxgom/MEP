"""Pygame adapters for distance and edge-weight heatmap rendering."""

from __future__ import annotations

import math


def draw_distance_colorbar(screen, has_scores, colorbar_bounds, canvas_top, canvas_height, heatmap_scale_mode, palette_index, heatmap_color, text_color):
    """Draw the active node-score palette legend."""
    import pygame

    if not has_scores:
        return
    colorbar_left, colorbar_width = colorbar_bounds
    bar_x, bar_y, bar_width, bar_height = colorbar_left + (colorbar_width - 20) // 2, canvas_top + 40, 20, canvas_height - 150
    pygame.draw.rect(screen, (255, 255, 255), (bar_x - 1, bar_y - 1, bar_width + 2, bar_height + 2), 1)
    for y in range(bar_height):
        t = 1.0 - y / bar_height
        color = heatmap_color(min(1.0, t / 0.75) if heatmap_scale_mode == 0 else t)
        pygame.draw.line(screen, color, (bar_x, bar_y + y), (bar_x + bar_width - 1, bar_y + y))
    font = pygame.font.SysFont("Outfit", 14, bold=True)
    high_label = font.render("HIGH", True, heatmap_color(1.0))
    low_label = font.render("LOW", True, heatmap_color(0.0))
    screen.blit(high_label, (bar_x + bar_width // 2 - high_label.get_width() // 2, bar_y - 18))
    screen.blit(low_label, (bar_x + bar_width // 2 - low_label.get_width() // 2, bar_y + bar_height + 6))
    title = font.render("VIRIDIS" if palette_index == 1 else "TURBO", True, text_color)
    screen.blit(title, (bar_x + bar_width // 2 - title.get_width() // 2, bar_y + bar_height + 24))


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


def draw_edge_weight_colorbar(screen, edge_weights, colorbar_bounds, canvas_top, canvas_height, block_weight, blocked_color, cool_colormap, log_scale, text_color):
    """Draw the edge-weight ratio legend and blocked-edge key."""
    import pygame

    if not edge_weights:
        return
    colorbar_left, colorbar_width = colorbar_bounds
    bar_x, bar_y, bar_width, bar_height = colorbar_left + (colorbar_width - 20) // 2, canvas_top + 40, 20, canvas_height - 150
    max_ratio, _log_max = log_scale(edge_weights, block_weight)
    pygame.draw.rect(screen, (255, 255, 255), (bar_x - 1, bar_y - 1, bar_width + 2, bar_height + 2), 1)
    for y in range(bar_height):
        color = cool_colormap(1.0 - y / bar_height)
        pygame.draw.line(screen, color, (bar_x, bar_y + y), (bar_x + bar_width - 1, bar_y + y))
    font = pygame.font.SysFont("Outfit", 13, bold=True)
    title = font.render("WGT", True, text_color)
    high_label = font.render(f"+{max_ratio:.1f}x", True, cool_colormap(1.0))
    low_label = font.render("+0x", True, cool_colormap(0.0))
    screen.blit(title, (bar_x + bar_width // 2 - title.get_width() // 2, bar_y - 32))
    screen.blit(high_label, (bar_x + bar_width // 2 - high_label.get_width() // 2, bar_y - 16))
    screen.blit(low_label, (bar_x + bar_width // 2 - low_label.get_width() // 2, bar_y + bar_height + 6))
    if any(value >= block_weight for value in edge_weights.values()):
        block_rect = pygame.Rect(bar_x, bar_y + bar_height + 28, bar_width, 8)
        pygame.draw.rect(screen, blocked_color, block_rect)
        blocked_label = font.render("BLOCK", True, text_color)
        screen.blit(blocked_label, (bar_x + bar_width // 2 - blocked_label.get_width() // 2, bar_y + bar_height + 40))
