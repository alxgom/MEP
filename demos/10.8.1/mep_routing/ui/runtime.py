"""Stateful canvas UI adapter for viewport, controls, overlays, and heatmaps."""

from __future__ import annotations

from dataclasses import dataclass, field

import pygame

from .colors import (
    cool_colormap,
    edge_weight_log_scale,
    heatmap_color,
    interpolate_regular_score,
    score_to_heatmap_t,
)
from .controls import (
    canvas_tool_button_bounds,
    draw_min_piece_slider,
    draw_weight_slider,
    slider_value_from_x,
    weight_view_switch_bounds,
)
from .drawing import draw_geometry_overlay, draw_polygon_hatch
from .heatmaps import (
    build_distance_heatmap_surface,
    draw_distance_colorbar,
    draw_edge_weight_colorbar,
    draw_edge_weight_heatmap,
)
from .canvas_tools import (
    draw_canvas_tool_controls,
    draw_ruler_overlay,
    draw_terminal_tool_buttons,
    terminal_tool_buttons,
)


@dataclass
class CanvasViewport:
    window_width: int = 1700
    window_height: int = 930
    canvas_left: int = 320
    canvas_top: int = 78
    panel_width: int = 280
    colorbar_width: int = 56
    world_width_mm: float = 15000.0
    world_height_mm: float = 11000.0
    zoom: float = 1.0
    pan_x_px: float = 0.0
    pan_y_px: float = 0.0
    canvas_width: int = field(init=False)
    canvas_height: int = field(init=False)
    colorbar_left: int = field(init=False)
    base_scale: float = field(init=False)
    scale: float = field(init=False)
    offset_x: float = field(init=False)
    offset_y: float = field(init=False)

    def __post_init__(self):
        self.update_layout(self.window_width, self.window_height)

    @property
    def canvas_rect(self):
        return self.canvas_left, self.canvas_top, self.canvas_width, self.canvas_height

    @property
    def window_size(self):
        return self.window_width, self.window_height

    def update_layout(self, width, height):
        self.window_width = max(1200, int(width))
        self.window_height = max(720, int(height))
        self.canvas_width = max(
            320,
            self.window_width - self.canvas_left - self.panel_width - self.colorbar_width - 10,
        )
        self.canvas_height = max(320, self.window_height - self.canvas_top - 40)
        self.colorbar_left = self.canvas_left + self.canvas_width + 10
        self.base_scale = min(
            self.canvas_width / self.world_width_mm,
            self.canvas_height / self.world_height_mm,
        )
        self._update_transform()

    def _update_transform(self):
        self.scale = self.base_scale * self.zoom
        center_x = self.canvas_left + self.canvas_width / 2.0
        center_y = self.canvas_top + self.canvas_height / 2.0
        self.offset_x = center_x - self.world_width_mm / 2.0 * self.scale + self.pan_x_px
        self.offset_y = center_y - self.world_height_mm / 2.0 * self.scale + self.pan_y_px

    def to_screen(self, x, y):
        return (
            int(self.offset_x + x * self.scale),
            int(self.offset_y + (self.world_height_mm - y) * self.scale),
        )

    def to_world(self, sx, sy):
        return (
            (sx - self.offset_x) / self.scale,
            self.world_height_mm - (sy - self.offset_y) / self.scale,
        )

    def set_zoom(self, value):
        self.zoom = max(0.5, min(6.0, float(value)))
        self._update_transform()

    def zoom_at(self, value, screen_pos):
        before = self.to_world(*screen_pos)
        self.set_zoom(value)
        after = self.to_screen(*before)
        self.pan_x_px += screen_pos[0] - after[0]
        self.pan_y_px += screen_pos[1] - after[1]
        self._update_transform()

    def pan_by(self, dx, dy):
        self.pan_x_px += dx
        self.pan_y_px += dy
        self._update_transform()

    def reset(self):
        self.zoom = 1.0
        self.pan_x_px = 0.0
        self.pan_y_px = 0.0
        self._update_transform()


@dataclass
class CanvasControlRects:
    min_piece: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 1, 1))
    bend: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 1, 1))
    crossing: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 1, 1))
    bend_reset: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 1, 1))
    crossing_reset: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 1, 1))
    terminal_tools: dict = field(default_factory=dict)


class CanvasUiRuntime:
    """Own mutable canvas-only state while domain choices remain in the app."""

    def __init__(self, viewport=None):
        self.viewport = viewport or CanvasViewport()
        self.controls = CanvasControlRects()
        self._heatmap_key = None
        self._heatmap_surface = None

    def invalidate_heatmap(self):
        self._heatmap_key = None
        self._heatmap_surface = None

    def update_layout(self, width, height):
        self.viewport.update_layout(width, height)
        self.invalidate_heatmap()

    def set_zoom(self, value):
        self.viewport.set_zoom(value)
        self.invalidate_heatmap()

    def zoom_at(self, value, screen_pos):
        self.viewport.zoom_at(value, screen_pos)
        self.invalidate_heatmap()

    def pan_by(self, dx, dy):
        self.viewport.pan_by(dx, dy)
        self.invalidate_heatmap()

    def reset_view(self):
        self.viewport.reset()
        self.invalidate_heatmap()

    def toolbar_buttons(self):
        return tuple(
            (action, pygame.Rect(bounds), label)
            for action, bounds, label in canvas_tool_button_bounds(
                self.viewport.canvas_left, self.viewport.canvas_top,
            )
        )

    def weight_switch_rect(self, weights_rect=None):
        if weights_rect is None:
            weights_rect = next(
                rect for action, rect, _label in self.toolbar_buttons() if action == "weights"
            )
        return pygame.Rect(weight_view_switch_bounds(tuple(weights_rect)))

    def terminal_buttons(self):
        return terminal_tool_buttons(
            self.viewport.canvas_left,
            self.viewport.canvas_top,
            self.viewport.canvas_width,
        )

    def apply_view_action(self, action):
        if action == "in":
            self.set_zoom(self.viewport.zoom * 1.25)
        elif action == "out":
            self.set_zoom(self.viewport.zoom / 1.25)
        elif action == "reset":
            self.reset_view()

    def slider_value(self, name, x, minimum, maximum, fallback):
        rect = getattr(self.controls, name)
        if rect.width <= 0:
            return fallback
        return slider_value_from_x(x, rect, minimum, maximum)

    def draw_min_piece_slider(self, screen, font, x, y, width, value, minimum, maximum, text, muted):
        self.controls.min_piece = draw_min_piece_slider(
            screen, font, x, y, width, value, minimum, maximum, text, muted,
        )

    def draw_weight_slider(
        self, screen, font, x, y, width, label, value, minimum, maximum,
        text, muted, *, name, suffix="", integer=False,
    ):
        rect, reset = draw_weight_slider(
            screen, font, x, y, width, label, value, minimum, maximum,
            text, muted, suffix=suffix, integer=integer,
        )
        setattr(self.controls, name, rect)
        setattr(self.controls, f"{name}_reset", reset)

    def draw_toolbar(self, screen, font, *, state):
        return draw_canvas_tool_controls(
            screen, font, self.toolbar_buttons(), self.weight_switch_rect(),
            ruler_enabled=state["ruler_enabled"],
            edge_weights_enabled=state["edge_weights_enabled"],
            diameter_width_enabled=state["diameter_width_enabled"],
            small_weight_view=state["small_weight_view"],
            zoom_level=self.viewport.zoom,
            dwelling_label=state["dwelling_label"],
            dwelling_options=state["dwelling_options"],
            dwelling_selector_open=state["dwelling_selector_open"],
            canvas_left=self.viewport.canvas_left,
            canvas_top=self.viewport.canvas_top,
            active_terminal_mode=state["active_terminal_mode"],
            text_color=state["text_color"],
        )

    def draw_terminal_tools(self, screen, font_bold, font_small, *, active_mode, overlay_enabled, colors):
        self.controls.terminal_tools = draw_terminal_tool_buttons(
            screen, font_bold, font_small, self.terminal_buttons(), active_mode, overlay_enabled,
            text_color=colors[0], muted_color=colors[1], allowed_color=colors[2],
        )

    def draw_ruler(self, screen, font, start, end, text_color):
        return draw_ruler_overlay(
            screen, font, start, end, self.viewport.to_screen, text_color=text_color,
        )

    def draw_geometry(self, screen, geometries, color):
        return draw_geometry_overlay(
            screen, geometries, color, self.viewport.to_screen, self.viewport.window_size,
        )

    def draw_hatch(self, screen, polygon, color, spacing=10, dashed=False):
        return draw_polygon_hatch(
            screen, polygon, color, self.viewport.to_screen, self.viewport.window_size,
            spacing, dashed,
        )

    def draw_distance_heatmap(
        self, screen, scores, *, base_env, grid_spacing, blocked_regions,
        scale_mode, palette_index, fill_color, hatch_color,
    ):
        if not scores or base_env is None:
            return
        key = (
            id(base_env), id(scores), len(scores), min(scores.values()), max(scores.values()),
            tuple(id(region) for region in blocked_regions), scale_mode, palette_index,
            self.viewport.canvas_width, self.viewport.canvas_height,
        )
        color = lambda value: heatmap_color(value, palette_index)
        if self._heatmap_key != key:
            interpolate = lambda x, y, grid: interpolate_regular_score(x, y, grid, grid_spacing)
            normalize = lambda value, low, high: score_to_heatmap_t(value, low, high, scale_mode)
            self._heatmap_surface = build_distance_heatmap_surface(
                scores, base_env.nodes, grid_spacing, self.viewport.canvas_rect,
                self.viewport.to_world, interpolate, normalize, color,
            )
            self._heatmap_key = key
        screen.blit(self._heatmap_surface, (self.viewport.canvas_left, self.viewport.canvas_top))
        self.draw_geometry(screen, blocked_regions, fill_color)
        for region in blocked_regions:
            self.draw_hatch(screen, region, hatch_color, spacing=9, dashed=True)

    def draw_distance_colorbar(self, screen, has_scores, scale_mode, palette_index, text_color):
        return draw_distance_colorbar(
            screen, has_scores, (self.viewport.colorbar_left, self.viewport.colorbar_width),
            self.viewport.canvas_top, self.viewport.canvas_height, scale_mode, palette_index,
            lambda value: heatmap_color(value, palette_index), text_color,
        )

    def draw_edge_heatmap(self, screen, overlay, env, block_weight, blocked_color):
        if not overlay or env is None:
            return
        return draw_edge_weight_heatmap(
            screen, overlay, env.nodes, env.adj, self.viewport.to_screen,
            block_weight, blocked_color, cool_colormap, edge_weight_log_scale,
        )

    def draw_edge_colorbar(self, screen, overlay, block_weight, blocked_color, text_color):
        if not overlay:
            return
        return draw_edge_weight_colorbar(
            screen, overlay, (self.viewport.colorbar_left, self.viewport.colorbar_width),
            self.viewport.canvas_top, self.viewport.canvas_height, block_weight, blocked_color,
            cool_colormap, edge_weight_log_scale, text_color,
        )
