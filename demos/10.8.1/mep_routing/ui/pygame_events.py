"""Pygame event adapter for the interactive routing workbench."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pygame
from shapely.geometry import Point, Polygon

from .events import (
    CanvasGestureState,
    CanvasHit,
    PanelHit,
    PanelInteractionState,
    begin_canvas_gesture,
    begin_panel_interaction,
    end_canvas_gesture,
    end_panel_interaction,
    move_canvas_gesture,
    move_panel_interaction,
)


@dataclass
class WorkbenchEventState:
    routes: object
    status: str
    elapsed_ms: float
    total_nodes: int
    running: bool = True
    selected_route_name: str | None = None
    dwelling_selector_open: bool = False
    needs_auto_placement: bool = False
    ruler_mode: bool = False
    last_wheel_rotate_ms: int = 0
    installation_drag_key: str | None = None
    installation_drag_moved: bool = False
    canvas_gesture: CanvasGestureState = CanvasGestureState()
    panel_interaction: PanelInteractionState = PanelInteractionState()

    @property
    def ruler_start_mm(self):
        return self.canvas_gesture.ruler_start_mm

    @property
    def ruler_end_mm(self):
        return self.canvas_gesture.ruler_end_mm

    @property
    def terminal_area_dragging(self):
        return self.canvas_gesture.terminal_area_dragging

    @property
    def terminal_area_start_mm(self):
        return self.canvas_gesture.terminal_area_start_mm

    @property
    def terminal_area_end_mm(self):
        return self.canvas_gesture.terminal_area_end_mm


class WorkbenchEventAdapter:
    """Own transient input state and execute one frame's Pygame events."""

    def __init__(self, app, screen, initial_run):
        self.app = app
        self.screen = screen
        self.state = WorkbenchEventState(
            *initial_run, needs_auto_placement=app.auto_placement_mode_idx > 0,
        )

    def _solve(self):
        result = self.app.solve_ventilation_routing()
        self.state.routes, self.state.status, self.state.elapsed_ms, self.state.total_nodes = result

    def _successful(self):
        return bool(self.state.routes) and not self.state.status.startswith("Blocked")

    def _record(self, label=None, color=(241, 196, 15)):
        if self._successful():
            self.app.record_current_solution(
                self.state.routes, self.state.elapsed_ms, label, color,
            )

    def _gesture(self):
        return replace(
            self.state.canvas_gesture,
            ruler_mode=self.state.ruler_mode,
            terminal_tool_mode=self.app.preferred_terminal_tool_mode,
        )

    def _cursor(self, active):
        self.app.set_ruler_cursor(active)

    def run_pending_auto_placement(self):
        if not self.state.needs_auto_placement:
            return
        self.state.needs_auto_placement = False
        self.app.run_auto_placement()
        self._solve()
        self._record("Auto", (230, 126, 34))

    def process(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.state.running = False
            elif event.type == pygame.VIDEORESIZE and not self.app.is_fullscreen:
                self.app.canvas_ui.update_layout(event.w, event.h)
                self.screen = pygame.display.set_mode(
                    self.app.canvas_ui.viewport.window_size, pygame.RESIZABLE,
                )
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._mouse_up(event)
            elif event.type == pygame.MOUSEMOTION:
                self._mouse_motion(event)
            elif event.type == pygame.KEYDOWN:
                self._key_down(event.key)
        return self.state

    def _mouse_down(self, event):
        if event.button == 1:
            self._left_down(event.pos)
        elif event.button == 2:
            transition = begin_canvas_gesture(
                self._gesture(), world_point=self.app.to_mm(*event.pos),
                screen_point=event.pos, shift=True, ctrl=False, hit=CanvasHit(),
            )
            self.state.canvas_gesture = transition.state
            try:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEALL)
            except pygame.error:
                pass
        elif event.button in (4, 5):
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                factor = 1.12 if event.button == 4 else 1 / 1.12
                self.app.canvas_ui.zoom_at(self.app.canvas_ui.viewport.zoom * factor, event.pos)
                return
            now = pygame.time.get_ticks()
            if now - self.state.last_wheel_rotate_ms < self.app.WHEEL_ROTATE_COOLDOWN_MS:
                return
            self.state.last_wheel_rotate_ms = now
            self.app.auto_placement_mode_idx = 0
            delta = 90 if event.button == 4 else -90
            self.app.machine_angle = (self.app.machine_angle + delta) % 360
            self._solve()
            self._record(f"Rot:{self.app.machine_angle}", (46, 204, 113))

    def _left_down(self, pos):
        a, s = self.app, self.state
        mx, my = pos
        installation_hit = a._installation_header_hit(a.get_installation_header_layout(), pos)
        if installation_hit is not None:
            kind, key = installation_hit
            if kind == "switch":
                a.installation_header_state, changed = a._toggle_installation(
                    a.installation_header_state, key, a.INSTALLATION_OPTIONS,
                )
                if changed:
                    self._solve()
                    s.selected_route_name = None
                else:
                    a.set_transient_message(f"{key.title()} is not available yet")
            else:
                s.installation_drag_key, s.installation_drag_moved = key, False
            return

        selector, options = a._dwelling_selector_bounds(
            a.canvas_ui.viewport.canvas_left, a.canvas_ui.viewport.canvas_top,
            len(a.dwelling_selector_options()),
        )
        if pygame.Rect(selector).collidepoint(pos):
            s.dwelling_selector_open = not s.dwelling_selector_open
            return
        selected = None
        if s.dwelling_selector_open:
            selected = next(
                (i for i, bounds in enumerate(options) if pygame.Rect(bounds).collidepoint(pos)), None,
            )
            s.dwelling_selector_open = False
        if selected is not None:
            a.dwelling_source_idx = a.DWELLING_SOURCE_MODES.index(
                "Random Synthetic" if selected == 0 else "Real DB"
            )
            if selected:
                a.real_scenario_idx = selected - 1
            self._change_dwelling(f"Home:{selected}", (52, 152, 219), clear_history=True)
            return

        help_card = next(
            (card for card, rect in a.help_button_rects.items() if rect.collidepoint(pos)), None,
        )
        controls = a.canvas_ui.controls
        panel_hit = PanelHit(
            help_card=help_card,
            min_piece_slider=controls.min_piece.collidepoint(pos),
            bend_slider=controls.bend.collidepoint(pos),
            bend_reset=controls.bend_reset.collidepoint(pos),
            crossing_slider=controls.crossing.collidepoint(pos),
            crossing_reset=controls.crossing_reset.collidepoint(pos),
        )
        occupied = any((panel_hit.help_card, panel_hit.min_piece_slider, panel_hit.bend_slider,
                        panel_hit.bend_reset, panel_hit.crossing_slider, panel_hit.crossing_reset))
        if not occupied:
            panel_hit = replace(panel_hit, canvas_tool=a.handle_canvas_tool_button_click(pos))
            occupied = panel_hit.canvas_tool is not None
        if not occupied:
            panel_hit = replace(panel_hit, solution_log_action=a.handle_solution_log_click(pos))
            occupied = panel_hit.solution_log_action is not None
        if not occupied:
            panel_hit = replace(panel_hit, terminal_tool_action=a.handle_terminal_tool_button_click(pos))
        panel = begin_panel_interaction(s.panel_interaction, hit=panel_hit, screen_x=mx)
        s.panel_interaction = panel.state
        if panel.commands:
            self._panel_command(panel.commands[0])
            return

        world = a.to_mm(mx, my)
        mods = pygame.key.get_mods()
        shift, ctrl = bool(mods & pygame.KMOD_SHIFT), bool(mods & pygame.KMOD_CTRL)
        hit, machine_center = CanvasHit(), None
        if not shift and not s.ruler_mode and a.preferred_terminal_tool_mode is None:
            pins = a.get_machine_pins(a.machine_cx, a.machine_cy, a.machine_angle)
            machine = Polygon([pins[name] for name in ("c_tl", "c_tr", "c_br", "c_bl")])
            point = Point(*world)
            machine_hit = machine.contains(point) or machine.distance(point) < 200.0
            route_names = {name for name, _ in s.routes} if s.routes else set()
            room_hit = a.find_room_route_at_point(world, route_names)
            route_hit = a.find_route_hit_at_point(s.routes, world)
            threshold = max(20.0, 4.0 / a.canvas_ui.viewport.scale)
            direct = route_hit[0] if route_hit and (not room_hit or route_hit[1] <= threshold) else None
            hit = CanvasHit(machine_hit, direct, room_hit)
            machine_center = (a.machine_cx, a.machine_cy) if machine_hit else None
        transition = begin_canvas_gesture(
            self._gesture(), world_point=world, screen_point=pos, shift=shift, ctrl=ctrl,
            hit=hit, machine_center_mm=machine_center,
        )
        s.canvas_gesture = transition.state
        for command in transition.commands:
            self._canvas_command(command)

    def _panel_command(self, command):
        a, s = self.app, self.state
        if command.name == "toggle_help":
            a.help_popup_card = None if a.help_popup_card == command.value else command.value
        elif command.name == "set_slider":
            name, x = command.value
            label, color = self._set_slider(name, x)
            self._solve(); self._record(label, color)
        elif command.name == "reset_slider":
            if command.value == "bend":
                a.reset_bend_weight(); marker = ("B:reset", (155, 89, 182))
            else:
                a.reset_crossing_weight(); marker = ("X:reset", (230, 126, 34))
            self._solve(); self._record(*marker)
        elif command.name == "canvas_tool":
            if command.value == "ruler":
                s.ruler_mode = not s.ruler_mode
                s.canvas_gesture = replace(self._gesture(), ruler_dragging=False)
                if s.ruler_mode:
                    a.preferred_terminal_tool_mode = None
                self._cursor(s.ruler_mode)
            elif command.value == "weights":
                a.edge_weight_heatmap_enabled = not a.edge_weight_heatmap_enabled
            elif command.value == "weight_view":
                a.edge_weight_view_mode_idx = (a.edge_weight_view_mode_idx + 1) % 2
        elif command.name == "solution_log":
            action = command.value
            if action == "log":
                a.log_current_solution(s.routes, s.status, s.elapsed_ms, s.total_nodes)
                return
            entry = (a.solution_log_session.auto_best_logs.get(action.split(":", 1)[1])
                     if isinstance(action, str) and action.startswith("best:") else
                     next((item for item in a.solution_log_session.manual_logs if item["id"] == action), None))
            if entry is not None:
                s.routes, s.status, s.elapsed_ms, s.total_nodes = a.restore_solution_log(entry)
                self._record(f"Back:L{action}", (255, 255, 255))
        elif command.name == "terminal_tool":
            if command.value == "reset":
                if a.terminal_runtime:
                    a.terminal_runtime.clear_preferences()
                self._solve(); self._record("Prefs reset", (26, 188, 156))
            if a.preferred_terminal_tool_mode:
                s.ruler_mode = False
                s.canvas_gesture = replace(self._gesture(), ruler_dragging=False)
            self._cursor(bool(a.preferred_terminal_tool_mode))

    def _set_slider(self, name, x):
        a = self.app
        if name == "min_piece":
            a.set_min_piece_factor_from_slider_x(x); return f"Min:{a.min_piece_factor:.2f}", (241, 196, 15)
        if name == "bend":
            a.set_bend_weight_from_slider_x(x); return f"B:{a.C_BEND:.0f}", (155, 89, 182)
        a.set_crossing_weight_from_slider_x(x); return f"X:{a.crossing_penalty_multiplier:.1f}", (230, 126, 34)

    def _canvas_command(self, command):
        a, s = self.app, self.state
        if command.name == "start_pan":
            try: pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEALL)
            except pygame.error: pass
        elif command.name == "apply_terminal_point":
            point, remove = command.value
            changed, room = a.apply_preferred_terminal_point(point, remove=remove)
            if room: s.selected_route_name = room
            elif not remove: a.set_transient_message("Invalid terminal: too close to wall or outside allowed room buffer")
            if changed: self._solve(); self._record("Term-" if remove else "Term+", (26, 188, 156))
        elif command.name == "apply_terminal_area":
            start, end, remove = command.value
            changed, room = a.apply_preferred_terminal_area(start, end, remove=remove)
            if room: s.selected_route_name = room
            if changed: self._solve(); self._record("Area-" if remove else "Area+", (155, 89, 182))
        elif command.name == "start_machine_drag": a.auto_placement_mode_idx = 0
        elif command.name == "select_route": s.selected_route_name = command.value
        elif command.name == "clear_route_selection": s.selected_route_name = None
        elif command.name == "pan_by": a.canvas_ui.pan_by(*command.value)
        elif command.name == "move_machine":
            a.machine_cx, a.machine_cy = command.value
            self._solve(); self._record()

    def _mouse_up(self, event):
        a, s = self.app, self.state
        if event.button not in (1, 2): return
        if event.button == 1 and s.installation_drag_key is not None:
            if not s.installation_drag_moved:
                a.installation_header_state, changed = a._activate_installation(
                    a.installation_header_state, s.installation_drag_key, a.INSTALLATION_OPTIONS,
                )
                if not changed and s.installation_drag_key != a.installation_header_state.active:
                    a.set_transient_message(f"{s.installation_drag_key.title()} is not available yet")
            s.installation_drag_key, s.installation_drag_moved = None, False
            return
        s.panel_interaction = end_panel_interaction(s.panel_interaction).state
        transition = end_canvas_gesture(self._gesture(), button="left" if event.button == 1 else "middle")
        s.canvas_gesture = transition.state
        for command in transition.commands: self._canvas_command(command)
        self._cursor(s.ruler_mode or bool(a.preferred_terminal_tool_mode))

    def _mouse_motion(self, event):
        a, s = self.app, self.state
        if s.installation_drag_key is not None:
            reordered = a._reorder_installation_at_x(
                a.installation_header_state, s.installation_drag_key, event.pos[0],
                a.get_installation_header_layout(),
            )
            if reordered.order != a.installation_header_state.order:
                a.installation_header_state, s.installation_drag_moved = reordered, True
            return
        panel = move_panel_interaction(s.panel_interaction, screen_x=event.pos[0])
        s.panel_interaction = panel.state
        if panel.commands:
            name, x = panel.commands[0].value
            self._set_slider(name, x); self._solve(); self._record(); return
        transition = move_canvas_gesture(
            self._gesture(), world_point=a.to_mm(*event.pos), screen_point=event.pos,
        )
        s.canvas_gesture = transition.state
        for command in transition.commands: self._canvas_command(command)

    def _change_dwelling(self, marker=None, color=(52, 152, 219), clear_history=False):
        a, s = self.app, self.state
        a.generate_new_dwelling(); a.solution_log_session.clear()
        if clear_history: a.clear_history_buffers()
        s.needs_auto_placement = a.auto_placement_mode_idx > 0
        self._solve(); self._record(marker, color)

    def _key_down(self, key):
        a, s = self.app, self.state
        if key == pygame.K_F11:
            a.is_fullscreen = not a.is_fullscreen
            if a.is_fullscreen:
                info = pygame.display.Info(); self.screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
                a.canvas_ui.update_layout(*self.screen.get_size())
            else:
                a.canvas_ui.update_layout(1700, 930); self.screen = pygame.display.set_mode(a.canvas_ui.viewport.window_size, pygame.RESIZABLE)
        elif key == pygame.K_ESCAPE:
            if s.ruler_mode:
                s.ruler_mode = False; s.canvas_gesture = replace(self._gesture(), ruler_dragging=False); self._cursor(False)
            elif a.preferred_terminal_tool_mode:
                a.preferred_terminal_tool_mode = None; s.canvas_gesture = replace(self._gesture(), terminal_area_dragging=False); self._cursor(False)
            else: s.selected_route_name = None
        elif key == pygame.K_SPACE:
            if a.DWELLING_SOURCE_MODES[a.dwelling_source_idx] == "Real DB":
                a.real_scenario_idx = (a.real_scenario_idx + 1) % len(a.REAL_DWELLING_SCENARIOS)
            self._change_dwelling(clear_history=True)
        elif transition := a.apply_routing_key_command(key):
            s.needs_auto_placement |= transition.needs_auto_placement
            self._solve()
            if transition.record_history and self._successful():
                a.record_history(s.routes, a.count_segment_crossings(s.routes), s.elapsed_ms)
                for marker in transition.markers: a.routing_history.add_marker(*marker)
        elif key == pygame.K_g: a.show_grid_graph = not a.show_grid_graph
        elif key == pygame.K_d:
            a.dwelling_source_idx = (a.dwelling_source_idx + 1) % len(a.DWELLING_SOURCE_MODES)
            self._change_dwelling(f"Src:{a.dwelling_source_idx}", (52, 152, 219), clear_history=True)
        elif key == pygame.K_o:
            a.routing_frame_idx = (a.routing_frame_idx + 1) % len(a.ROUTING_FRAME_OPTIONS)
            self._change_dwelling(f"Frame:{a.routing_frame_idx}", (230, 126, 34))
        elif key == pygame.K_v:
            a.show_heatmap = not a.show_heatmap
            if a.show_heatmap: a.ensure_placement_heatmap_scores()
        elif key == pygame.K_m: a.edge_weight_heatmap_enabled = not a.edge_weight_heatmap_enabled
        elif key == pygame.K_n: a.edge_weight_view_mode_idx = (a.edge_weight_view_mode_idx + 1) % 2
        elif key == pygame.K_x: a.route_real_diameter_width_enabled = not a.route_real_diameter_width_enabled
        elif key == pygame.K_h:
            a.heatmap_scale_mode = (a.heatmap_scale_mode + 1) % 2
            self._record(f"H:{'Log' if a.heatmap_scale_mode else 'Lin'}", (150, 150, 150))
        elif key == pygame.K_b:
            a.heatmap_palette_idx = (a.heatmap_palette_idx + 1) % 2
            self._record(f"Pal:{'Vir' if a.heatmap_palette_idx else 'Tur'}", (26, 188, 156))
