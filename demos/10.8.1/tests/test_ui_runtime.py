from mep_routing.ui.runtime import CanvasViewport


def test_canvas_viewport_round_trips_world_coordinates():
    viewport = CanvasViewport()

    point = (4321.0, 6789.0)
    screen = viewport.to_screen(*point)
    restored = viewport.to_world(*screen)

    assert abs(restored[0] - point[0]) <= 1.0 / viewport.scale
    assert abs(restored[1] - point[1]) <= 1.0 / viewport.scale


def test_zoom_at_keeps_world_point_under_cursor():
    viewport = CanvasViewport()
    cursor = (viewport.canvas_left + 217, viewport.canvas_top + 193)
    before = viewport.to_world(*cursor)

    viewport.zoom_at(2.0, cursor)
    after = viewport.to_world(*cursor)

    assert abs(after[0] - before[0]) <= 1.0 / viewport.scale
    assert abs(after[1] - before[1]) <= 1.0 / viewport.scale


def test_layout_enforces_minimum_window_and_canvas_sizes():
    viewport = CanvasViewport()

    viewport.update_layout(100, 100)

    assert viewport.window_size == (1200, 720)
    assert viewport.canvas_width >= 320
    assert viewport.canvas_height >= 320
