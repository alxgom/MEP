from mep_routing.ui.heatmaps import horizontal_colorbar_bounds


def test_horizontal_colorbar_uses_bottom_right_canvas_space():
    bounds = horizontal_colorbar_bounds((320, 78, 1090, 812))

    assert bounds == (1152, 836, 240, 40)


def test_horizontal_colorbar_contract_shrinks_for_narrow_canvases():
    x, y, width, height = horizontal_colorbar_bounds((10, 20, 160, 300))

    assert (x, y, width, height) == (12, 266, 140, 40)
