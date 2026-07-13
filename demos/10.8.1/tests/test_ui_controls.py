from mep_routing.ui.controls import (
    canvas_tool_button_bounds,
    slider_fraction,
    slider_value_from_x,
    weight_view_switch_bounds,
)


class Rect:
    def __init__(self, x, width):
        self.x = x
        self.width = width


def test_canvas_tool_button_bounds_preserve_toolbar_order_and_spacing():
    buttons = canvas_tool_button_bounds(100, 50)

    assert [action for action, _bounds, _label in buttons] == ["in", "out", "reset", "ruler", "weights", "diameter"]
    assert buttons[0][1] == (112, 62, 28, 28)
    assert buttons[4][1] == (296, 62, 72, 28)
    assert buttons[5][1] == (472, 62, 54, 28)


def test_weight_view_switch_bounds_follows_weights_button():
    assert weight_view_switch_bounds((296, 62, 72, 28)) == (374, 64, 92, 24)


def test_slider_value_from_x_clamps_and_handles_empty_rectangles():
    rect = Rect(10, 100)

    assert slider_value_from_x(-10, rect, 2, 6) == 2
    assert slider_value_from_x(60, rect, 2, 6) == 4
    assert slider_value_from_x(300, rect, 2, 6) == 6
    assert slider_value_from_x(20, Rect(0, 0), 2, 6) == 2


def test_slider_fraction_clamps_invalid_and_out_of_range_values():
    assert slider_fraction(2, 2, 6) == 0
    assert slider_fraction(4, 2, 6) == 0.5
    assert slider_fraction(10, 2, 6) == 1
    assert slider_fraction(4, 2, 2) == 0
