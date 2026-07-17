"""Pygame rendering for the interactive router dashboard sidebar.

The renderer receives already-derived display values.  This keeps routing,
placement, and application-state ownership outside the Pygame drawing layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .dropdown import DropdownOption, DropdownState, draw_dropdown, dropdown_bounds


Color = tuple[int, int, int]
MACHINE_SELECTOR_Y = 108
CLI_RETURN_TOGGLE_BOUNDS = (176, 247, 94, 20)
SOLVER_DROPDOWN_X = 105
SOLVER_DROPDOWN_Y = 306
SOLVER_DROPDOWN_WIDTH = 175
SOLVER_DROPDOWN_HEIGHT = 22
SOLVER_DROPDOWN_STEP = 26
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
    status_surface: Color = (33, 37, 46)
    success: Color = (46, 204, 113)
    failure: Color = (231, 76, 60)


@dataclass(frozen=True)
class AutoPlacementCard:
    mode: str
    heatmap: str
    placement_weights: str
    rotation_mode: str
    shortcuts: str = "[A] Auto | [P] Mode | [V] Heatmap | [?] More"


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
    dropdowns: tuple["SolverDropdown", ...] = ()


@dataclass(frozen=True)
class SolverDropdown:
    key: str
    label: str
    options: tuple[DropdownOption, ...]
    selected_key: str
    open: bool = False


def solver_dropdown_bounds(index: int, option_count: int):
    return dropdown_bounds(
        SOLVER_DROPDOWN_X,
        SOLVER_DROPDOWN_Y + index * SOLVER_DROPDOWN_STEP,
        SOLVER_DROPDOWN_WIDTH,
        option_count,
        height=SOLVER_DROPDOWN_HEIGHT,
    )


def dropdown_overlay_order(indexed_dropdowns):
    """Draw the open control last so its option sheet remains a true overlay."""
    return tuple(item for item in indexed_dropdowns if not item[1].open) + tuple(
        item for item in indexed_dropdowns if item[1].open
    )


@dataclass(frozen=True)
class MachineCard:
    frame: str
    position_mm: tuple[float, float] | None
    rotation: str
    machine_options: tuple[DropdownOption, ...] = ()
    selected_machine_key: str = ""
    machine_selector_open: bool = False
    return_ventilation_enabled: bool | None = None


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
    if view.machine.return_ventilation_enabled is None:
        screen.blit(fonts.small.render(view.auto_placement.shortcuts, True, colors.muted), (text_x, 252))
    else:
        screen.blit(fonts.small.render("Return ventilation", True, colors.muted), (text_x, 251))
        toggle = pygame.Rect(CLI_RETURN_TOGGLE_BOUNDS)
        fill = (52, 152, 219) if view.machine.return_ventilation_enabled else (70, 74, 82)
        pygame.draw.rect(screen, fill, toggle, border_radius=toggle.height // 2)
        label = fonts.small.render("ON" if view.machine.return_ventilation_enabled else "OFF", True, colors.text)
        screen.blit(label, (toggle.centerx - label.get_width() // 2, toggle.centery - label.get_height() // 2))

    solver_card = pygame.Rect(15, 280, card_width, 270)
    pygame.draw.rect(screen, colors.card, solver_card, border_radius=6)
    screen.blit(fonts.bold.render("ROUTING PATH SOLVER", True, colors.text), (text_x, 290))
    draw_help_button("solver", solver_card, fonts.small)
    for index, dropdown in enumerate(view.solver.dropdowns):
        y = SOLVER_DROPDOWN_Y + index * SOLVER_DROPDOWN_STEP + 3
        screen.blit(fonts.small.render(dropdown.label, True, colors.muted), (text_x, y))
    screen.blit(
        fonts.small.render(
            format_selected_route(
                view.solver.selected_route, view.solver.preferred_terminal_count,
            ),
            True, colors.text,
        ),
        (text_x, 440),
    )
    draw_min_piece_slider(screen, fonts.small, text_x, 460, canvas_left - 70)
    draw_weight_slider(
        screen, fonts.small, text_x, 495, canvas_left - 70, "Bend weight",
        view.solver.bend_value, view.solver.bend_min, view.solver.bend_max,
        (155, 89, 182), "bend", "", True,
    )
    draw_weight_slider(
        screen, fonts.small, text_x, 522, canvas_left - 70, "Cross x bend",
        view.solver.crossing_value, view.solver.crossing_min, view.solver.crossing_max,
        (230, 126, 34), "crossing", "x", False,
    )

    status_y = 560
    status_card = pygame.Rect(15, status_y, card_width, 170)
    pygame.draw.rect(screen, colors.status_surface, status_card, border_radius=4)
    pygame.draw.rect(screen, (55, 61, 72), status_card, 1, border_radius=4)
    screen.blit(fonts.bold.render("SOLVER EXECUTION STATUS", True, colors.muted), (text_x, status_y + 10))
    message = view.execution.message.strip()
    normalized = message.lower()
    semantic = None
    if normalized.startswith("success"):
        semantic, message = ("SUCCESS", colors.success), message.partition(":")[2].strip()
    elif normalized.startswith(("fail", "error", "routing blocked")):
        semantic = ("FAILED", colors.failure)
    detail_y = status_y + 30
    if semantic is not None:
        screen.blit(fonts.bold.render(semantic[0], True, semantic[1]), (text_x, detail_y))
        detail_y += max(18, fonts.bold.get_height() + 2)
    for index, line in enumerate(wrap_status_lines(message, max_line_length=34)):
        screen.blit(fonts.small.render(line, True, colors.text), (text_x, detail_y + index * 17))

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
    indexed_dropdowns = tuple(enumerate(view.solver.dropdowns))
    for index, dropdown in dropdown_overlay_order(indexed_dropdowns):
        control, _option_bounds = solver_dropdown_bounds(index, len(dropdown.options))
        draw_dropdown(
            screen, fonts.small, bounds=control,
            options=dropdown.options,
            state=DropdownState(dropdown.selected_key, dropdown.open),
            text_color=colors.text, muted_color=colors.muted,
        )
