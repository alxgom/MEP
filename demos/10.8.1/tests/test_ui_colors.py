from vent_router.ui import (
    cool_colormap,
    heatmap_color,
    score_to_heatmap_t,
    turbo_color,
    viridis_color,
)


def test_turbo_color_clamps_to_palette_endpoints():
    assert turbo_color(-1.0) == (48, 18, 59)
    assert turbo_color(2.0) == (122, 4, 3)


def test_viridis_color_interpolates_between_control_points():
    assert viridis_color(0.075) == (69, 22, 103)


def test_heatmap_color_selects_palette_by_index():
    assert heatmap_color(0.0, palette_idx=0) == turbo_color(0.0)
    assert heatmap_color(0.0, palette_idx=1) == viridis_color(0.0)


def test_score_to_heatmap_t_supports_saturated_linear_scale():
    assert score_to_heatmap_t(75, 0, 100, scale_mode=0) == 1.0
    assert score_to_heatmap_t(37.5, 0, 100, scale_mode=0) == 0.5


def test_score_to_heatmap_t_supports_log_scale():
    assert score_to_heatmap_t(1, 1, 100, scale_mode=1) == 0.0
    assert score_to_heatmap_t(100, 1, 100, scale_mode=1) == 1.0


def test_cool_colormap_clamps_and_interpolates():
    assert cool_colormap(-1.0) == (0, 255, 255)
    assert cool_colormap(0.5) == (127, 127, 255)
    assert cool_colormap(2.0) == (255, 0, 255)
