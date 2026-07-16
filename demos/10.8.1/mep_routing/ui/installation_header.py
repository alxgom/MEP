"""Installation routing-order header and interaction state."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class InstallationOption:
    key: str
    label: str
    available: bool = False


@dataclass(frozen=True)
class InstallationHeaderState:
    order: tuple[str, ...]
    enabled: frozenset[str]
    active: str


@dataclass(frozen=True)
class InstallationHeaderLayout:
    pill_bounds: dict[str, tuple[int, int, int, int]]
    switch_bounds: dict[str, tuple[int, int, int, int]]
    arrow_segments: tuple[tuple[tuple[int, int], tuple[int, int]], ...]


def installation_header_layout(canvas_left, header_top, order, *, pill_width=126, pill_height=40, gap=42):
    pill_bounds = {}
    switch_bounds = {}
    arrow_segments = []
    x = int(canvas_left) + 14
    y = int(header_top) + 16
    for index, key in enumerate(order):
        pill_bounds[key] = (x, y, pill_width, pill_height)
        switch_bounds[key] = (x + 8, y + 8, 38, pill_height - 16)
        if index < len(order) - 1:
            start_x = x + pill_width + 8
            end_x = x + pill_width + gap - 8
            mid_y = y + pill_height // 2
            arrow_segments.extend((
                ((start_x, mid_y), (end_x, mid_y)),
                ((end_x - 7, mid_y - 6), (end_x, mid_y)),
                ((end_x - 7, mid_y + 6), (end_x, mid_y)),
            ))
        x += pill_width + gap
    return InstallationHeaderLayout(pill_bounds, switch_bounds, tuple(arrow_segments))


def installation_header_hit(layout, point):
    px, py = point
    for key, bounds in layout.switch_bounds.items():
        x, y, width, height = bounds
        if x <= px < x + width and y <= py < y + height:
            return "switch", key
    for key, bounds in layout.pill_bounds.items():
        x, y, width, height = bounds
        if x <= px < x + width and y <= py < y + height:
            return "pill", key
    return None


def toggle_installation(state, key, options):
    option = next((item for item in options if item.key == key), None)
    if option is None or not option.available:
        return state, False
    enabled = set(state.enabled)
    if key in enabled:
        enabled.remove(key)
    else:
        enabled.add(key)
    return replace(state, enabled=frozenset(enabled)), True


def activate_installation(state, key, options):
    option = next((item for item in options if item.key == key), None)
    if option is None or not option.available:
        return state, False
    return replace(state, active=key), state.active != key


def move_installation(state, key, target_index):
    if key not in state.order:
        return state
    order = list(state.order)
    old_index = order.index(key)
    order.pop(old_index)
    target_index = max(0, min(int(target_index), len(order)))
    order.insert(target_index, key)
    return replace(state, order=tuple(order))


def reorder_installation_at_x(state, key, screen_x, layout):
    centers = [
        (other, bounds[0] + bounds[2] / 2.0)
        for other, bounds in layout.pill_bounds.items()
        if other != key
    ]
    target = sum(1 for _other, center in centers if screen_x > center)
    return move_installation(state, key, target)


def draw_installation_header(screen, font, small_font, state, options, layout, *, dragged_key=None):
    import pygame

    option_by_key = {option.key: option for option in options}
    title = small_font.render(
        f"ROUTING ORDER  |  active: {option_by_key[state.active].label}", True, (150, 158, 170),
    )
    first_x = min(bounds[0] for bounds in layout.pill_bounds.values())
    first_y = min(bounds[1] for bounds in layout.pill_bounds.values())
    screen.blit(title, (first_x, first_y - title.get_height() - 2))

    for start, end in layout.arrow_segments:
        pygame.draw.line(screen, (130, 138, 148), start, end, 2)

    for key in state.order:
        option = option_by_key[key]
        rect = pygame.Rect(layout.pill_bounds[key])
        active = key == state.active
        enabled = key in state.enabled
        fill = (58, 76, 104) if active else (48, 52, 60)
        if active and enabled:
            fill = (56, 86, 130)
        if not option.available:
            fill = (43, 45, 50)
        border = (91, 149, 245) if active else (122, 128, 138)
        if dragged_key == key:
            border = (241, 196, 15)
        pygame.draw.rect(screen, fill, rect, border_radius=rect.height // 2)
        pygame.draw.rect(screen, border, rect, 2 if active or dragged_key == key else 1, border_radius=rect.height // 2)

        switch = pygame.Rect(layout.switch_bounds[key])
        switch_fill = (92, 150, 245) if enabled else (70, 74, 82)
        if not option.available:
            switch_fill = (58, 60, 66)
        pygame.draw.rect(screen, switch_fill, switch, border_radius=switch.height // 2)
        knob_radius = switch.height // 2 - 2
        knob_x = switch.right - switch.height // 2 if enabled else switch.left + switch.height // 2
        pygame.draw.circle(screen, (238, 240, 244), (knob_x, switch.centery), knob_radius)

        text_color = (225, 229, 236) if option.available else (112, 116, 124)
        label = font.render(option.label, True, text_color)
        label_x = switch.right + (rect.right - switch.right - label.get_width()) // 2
        screen.blit(label, (label_x, rect.centery - label.get_height() // 2))
