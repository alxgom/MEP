"""Stateful Pygame chrome surrounding the routing canvas."""

from __future__ import annotations


class WorkbenchChromeRuntime:
    def __init__(self, viewport, terminal_ui, panel_width, colors):
        self.viewport = viewport
        self.terminal_ui = terminal_ui
        self.panel_width = panel_width
        self.colors = colors
        self.help_popup_card = None
        self.help_button_rects = {}
        self.log_button_rect = None
        self.log_row_rects = []
        self.transient_message = None
        self.transient_message_until_ms = 0

    def toggle_help(self, card_id):
        self.help_popup_card = None if self.help_popup_card == card_id else card_id

    def set_transient_message(self, text, now_ms, duration_ms=2400):
        self.transient_message = str(text)
        self.transient_message_until_ms = int(now_ms) + int(duration_ms)

    def transient_message_at(self, now_ms):
        if not self.transient_message or int(now_ms) > self.transient_message_until_ms:
            return None
        return self.transient_message

    def solution_log_action(self, pos):
        from .solution_logs import solution_log_action
        if self.log_button_rect is None:
            return None
        return solution_log_action(pos, self.log_button_rect, self.log_row_rects)

    def draw_solution_logs(self, screen, font_small, font_bold, logs):
        from .solution_logs import draw_solution_logs_panel
        self.log_button_rect, self.log_row_rects = draw_solution_logs_panel(
            screen, font_small, font_bold,
            window_width=self.viewport.window_width,
            window_height=self.viewport.window_height,
            panel_width=self.panel_width,
            solution_logs=logs.manual_logs,
            auto_best_logs=logs.auto_best_logs,
            selected_log_id=logs.selected_log_id,
            plot_background_color=self.colors["plot_background"],
            text_color=self.colors["text"], muted_color=self.colors["muted"],
        )

    def draw_help_button(self, screen, card_id, rect, font):
        from .help import draw_card_help_button
        self.help_button_rects[card_id] = draw_card_help_button(
            screen, card_id, rect, font, self.help_popup_card == card_id,
            self.colors["muted"], self.colors["text"],
        )

    def draw_help_popup(self, screen, font):
        from .help import draw_help_popup, help_lines
        return draw_help_popup(
            screen, font, help_lines(self.help_popup_card),
            (self.viewport.canvas_left + 16, self.viewport.canvas_top + 58),
            self.colors["text"],
        )

    def show_message(self, text, duration_ms=2400):
        import pygame
        self.set_transient_message(text, pygame.time.get_ticks(), duration_ms)

    def draw_transient(self, screen, font):
        import pygame
        from .help import draw_transient_message
        message = self.transient_message_at(pygame.time.get_ticks())
        if message is not None:
            return draw_transient_message(
                screen, font, message,
                (self.viewport.canvas_left, self.viewport.canvas_top),
                self.colors["text"],
            )

    def draw_viewer_legend(self, screen, font, show_terminal_validity):
        from .help import draw_viewer_legend
        return draw_viewer_legend(
            screen, font, self.viewport.canvas_rect, show_terminal_validity,
            self.colors["plan_label"], self.colors["wet_room"], self.colors["wall"],
            self.terminal_ui.draw_validity_square,
        )
