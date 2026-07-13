from mep_routing.ui import (
    cool_colormap,
    edge_weight_log_scale,
    heatmap_color,
    interpolate_regular_score,
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


def test_interpolate_regular_score_uses_bilinear_then_nearby_fallback():
    score_grid = {(0, 0): 0.0, (1, 0): 10.0, (0, 1): 20.0, (1, 1): 30.0}

    assert interpolate_regular_score(50.0, 50.0, score_grid, 100.0) == 15.0
    assert interpolate_regular_score(0.0, 0.0, {(0, 0): 5.0}, 100.0) == 5.0


def test_edge_weight_log_scale_excludes_blocked_edges():
    max_ratio, log_max = edge_weight_log_scale({(0, 1): 3.0, (1, 2): 1000.0}, block_weight=1000.0)

    assert max_ratio == 3.0
    assert log_max > 0.0
