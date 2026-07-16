"""Pure selection state/layout plus a thin Pygame dropdown renderer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence


Bounds = tuple[int, int, int, int]


@dataclass(frozen=True)
class DropdownOption:
    key: str
    label: str
    detail: str = ""


@dataclass(frozen=True)
class DropdownState:
    selected_key: str
    open: bool = False


@dataclass(frozen=True)
class DropdownTransition:
    state: DropdownState
    changed: bool = False


def dropdown_transition(state: DropdownState, action: str, key: str | None = None) -> DropdownTransition:
    if action == "toggle":
        return DropdownTransition(replace(state, open=not state.open))
    if action == "dismiss":
        return DropdownTransition(replace(state, open=False))
    if action == "choose":
        if key is None:
            raise ValueError("choose requires an option key")
        return DropdownTransition(DropdownState(key, False), key != state.selected_key)
    raise ValueError(f"unknown dropdown action: {action}")


def dropdown_bounds(x: int, y: int, width: int, option_count: int, *, height: int = 24, gap: int = 2):
    control = (x, y, width, height)
    options = tuple((x, y + height + gap + i * height, width, height) for i in range(option_count))
    return control, options


def draw_dropdown(screen, font, *, bounds: Bounds, options: Sequence[DropdownOption], state: DropdownState,
                  text_color=(236, 240, 241), muted_color=(149, 165, 166)) -> None:
    """Draw a dropdown. Callers retain hit testing and application side effects."""
    import pygame

    if len(options) <= 1:
        return
    control_bounds, option_bounds = dropdown_bounds(*bounds[:3], len(options), height=bounds[3])
    control = pygame.Rect(control_bounds)
    selected = next(option for option in options if option.key == state.selected_key)
    pygame.draw.rect(screen, (32, 36, 44), control, border_radius=4)
    pygame.draw.rect(screen, (150, 158, 166), control, 1, border_radius=4)
    label = font.render(selected.label[:28], True, text_color)
    screen.blit(label, (control.x + 8, control.centery - label.get_height() // 2))
    arrow = font.render("^" if state.open else "v", True, muted_color)
    screen.blit(arrow, (control.right - arrow.get_width() - 8, control.centery - arrow.get_height() // 2))
    if state.open:
        for option, raw_bounds in zip(options, option_bounds):
            rect = pygame.Rect(raw_bounds)
            fill = (58, 80, 94) if option.key == state.selected_key else (32, 36, 44)
            pygame.draw.rect(screen, fill, rect)
            pygame.draw.rect(screen, (95, 104, 112), rect, 1)
            text = f"{option.label}  {option.detail}".strip()
            rendered = font.render(text[:32], True, text_color)
            screen.blit(rendered, (rect.x + 8, rect.centery - rendered.get_height() // 2))
