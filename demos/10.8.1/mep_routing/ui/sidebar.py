"""Pygame rendering for the interactive router dashboard sidebar.

The renderer receives already-derived display values.  This keeps routing,
placement, and application-state ownership outside the Pygame drawing layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .dropdown import DropdownOption, DropdownState, draw_dropdown


Color = tuple[int, int, int]
MACHINE_SELECTOR_Y = 108
HelpButtonDrawer = Callable[[str, Any, Any], None]
MinPieceSliderDrawer = Callable[[Any, Any, int, int, int], None]
WeightSliderDrawer = Callable[[Any, Any, int, int, int, str, float, float, float, Color, str, str, bool], None]


@dataclass(frozen=True)
class SidebarFonts:
    title: Any
    bold: Any
    small: Any


@dataclass(frozen=True)
class SidebarColors:
    panel: Color
    wall: Color
    text: Color
    muted: Color
    warning: Color = (241, 196, 15)
    card: Color = (40, 45, 55)


@dataclass(frozen=True)
class AutoPlacementCard:
    mode: str
    heatmap: str
    placement_weights: str
    rotation_mode: str


@dataclass(frozen=True)
class SolverCard:
    strategy: str
    router: str
    heuristic: str
    grid_type: str
    starts: str
    selected_route: str
    preferred_terminal_count: int
    bend_value: float
    bend_min: float
    bend_max: float
    crossing_value: float
    crossing_min: float
    crossing_max: float


@dataclass(frozen=True)
class MachineCard:
    frame: str
    position_mm: tuple[float, float] | None
    rotation: str
    machine_options: tuple[DropdownOption, ...] = ()
    selected_machine_key: str = ""
    machine_selector_open: bool = False


@dataclass(frozen=True)
class ExecutionStatusCard:
    message: str
    validation_warnings: Sequence[str]
    elapsed_ms: float
    total_nodes: int
    fps: float


@dataclass(frozen=True)
class SidebarView:
    auto_placement: AutoPlacementCard
    solver: SolverCard
    machine: MachineCard
    execution: ExecutionStatusCard


def wrap_status_lines(message: str, max_line_length: int = 28) -> tuple[str, ...]:
    """Wrap a status message by words for the fixed-width status card."""
    if max_line_length < 1:
        raise ValueError("max_line_length must be positive")

    lines: list[str] = []
    current = ""
    for word in message.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_line_length:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return tuple(lines)


def format_validation_warning(
    warnings: Sequence[str],
    *,
    max_warnings: int = 2,
    max_length: int = 42,
) -> str:
    """Return the compact warning summary displayed in the status card."""
    if max_warnings < 1:
        raise ValueError("max_warnings must be positive")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if not warnings:
        return "Warnings: none"[:max_length]

    text = "Warnings: " + ", ".join(warnings[:max_warnings])
    if len(warnings) > max_warnings:
        text += f" +{len(warnings) - max_warnings}"
    return text[:max_length]


def format_selected_route(selected_route: str | None, preferred_terminal_count: int) -> str:
    """Format the selected route summary with the legacy card truncation."""
    selected = selected_route or "None"
    return f"Selected: {selected[:14]} | Prefs: {preferred_terminal_count}"


def draw_sidebar(
    screen: Any,
    *,
    canvas_left: int,
    window_height: int,
    fonts: SidebarFonts,
    colors: SidebarColors,
    view: SidebarView,
    draw_help_button: HelpButtonDrawer,
    draw_min_piece_slider: MinPieceSliderDrawer,
    draw_weight_slider: WeightSliderDrawer,
) -> None:
    """Draw the dashboard sidebar while leaving hit-rectangle ownership to callers."""
    import pygame

    panel_width = canvas_left - 10
    card_width = canvas_left - 40
    text_x = 25

    pygame.draw.rect(screen, colors.panel, (0, 0, panel_width, window_height))
    pygame.draw.line(screen, colors.wall, (panel_width, 0), (panel_width, window_height), 2)

    screen.blit(fonts.title.render("Auto-Placement visualizer", True, colors.text), (20, 20))
    screen.blit(fonts.small.render("Vents & Extraction Router Dashboard", True, colors.muted), (20, 42))

    auto_card = pygame.Rect(15, 75, card_width, 195)
    pygame.draw.rect(screen, colors.card, auto_card, border_radius=6)
    screen.blit(fonts.bold.render("MACHINE & AUTO-PLACEMENT", True, colors.text), (text_x, 85))
    draw_help_button("auto", auto_card, fonts.small)
    if view.machine.position_mm is None:
        position_text = "Placement: interactive runtime pending"
    else:
        machine_x, machine_y = view.machine.position_mm
        position_text = f"Position: ({int(machine_x)}, {int(machine_y)}) mm"
    screen.blit(fonts.small.render(position_text, True, colors.text), (text_x, 138))
    screen.blit(fonts.small.render(f"Frame: {view.machine.frame[:24]}", True, colors.muted), (text_x, 156))
    screen.blit(fonts.bold.render(f"Mode: {view.auto_placement.mode}", True, colors.text), (text_x, 178))
    screen.blit(fonts.small.render(f"Heatmap: {view.auto_placement.heatmap}", True, colors.muted), (text_x, 198))
    screen.blit(
        fonts.small.render(f"Weights: {view.auto_placement.placement_weights}", True, colors.muted),
        (text_x, 216),
    )
    screen.blit(fonts.small.render(view.machine.rotation[:38], True, colors.text), (text_x, 234))
    screen.blit(fonts.small.render("[A] Auto | [P] Mode | [V] Heatmap | [?] More", True, colors.muted), (text_x, 252))

    solver_card = pygame.Rect(15, 280, card_width, 230)
    pygame.draw.rect(screen, colors.card, solver_card, border_radius=6)
    screen.blit(fonts.bold.render("ROUTING PATH SOLVER", True, colors.text), (text_x, 290))
    draw_help_button("solver", solver_card, fonts.small)
    solver_rows = (
        (f"Strategy: {view.solver.strategy}", 310),
        (f"Router: {view.solver.router}", 330),
        (f"Heuristic: {view.solver.heuristic}", 350),
        (f"Grid type: {view.solver.grid_type}", 370),
        (f"Starts: {view.solver.starts}", 390),
        (format_selected_route(view.solver.selected_route, view.solver.preferred_terminal_count), 410),
    )
    for label, y in solver_rows:
        screen.blit(fonts.small.render(label, True, colors.text), (text_x, y))
    draw_min_piece_slider(screen, fonts.small, text_x, 430, canvas_left - 70)
    draw_weight_slider(
        screen, fonts.small, text_x, 465, canvas_left - 70, "Bend weight",
        view.solver.bend_value, view.solver.bend_min, view.solver.bend_max,
        (155, 89, 182), "bend", "", True,
    )
    draw_weight_slider(
        screen, fonts.small, text_x, 492, canvas_left - 70, "Cross x bend",
        view.solver.crossing_value, view.solver.crossing_min, view.solver.crossing_max,
        (230, 126, 34), "crossing", "x", False,
    )

    status_y = 520
    status_card = pygame.Rect(15, status_y, card_width, 170)
    pygame.draw.rect(screen, colors.card, status_card, border_radius=6)
    screen.blit(fonts.bold.render("SOLVER EXECUTION STATUS", True, colors.text), (text_x, status_y + 10))
    draw_help_button("status", status_card, fonts.small)
    for index, line in enumerate(wrap_status_lines(view.execution.message)):
        screen.blit(fonts.small.render(line, True, colors.text), (text_x, status_y + 30 + index * 18))

    warning_text = format_validation_warning(view.execution.validation_warnings)
    warning_color = colors.muted if not view.execution.validation_warnings else colors.warning
    screen.blit(fonts.small.render(warning_text, True, warning_color), (text_x, status_y + 95))
    screen.blit(fonts.small.render(f"Pathfinder time: {view.execution.elapsed_ms:.1f} ms", True, colors.text), (text_x, status_y + 125))
    screen.blit(fonts.small.render(f"Total routed nodes: {view.execution.total_nodes}", True, colors.muted), (text_x, status_y + 145))
    screen.blit(fonts.small.render(f"Render engine: Pygame ({view.execution.fps:.0f} FPS)", True, colors.muted), (text_x, status_y + 165))
    if view.machine.machine_options:
        draw_dropdown(
            screen, fonts.small, bounds=(text_x, MACHINE_SELECTOR_Y, canvas_left - 70, 24),
            options=view.machine.machine_options,
            state=DropdownState(view.machine.selected_machine_key, view.machine.machine_selector_open),
            text_color=colors.text, muted_color=colors.muted,
        )
