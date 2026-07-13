from __future__ import annotations

import math


TURBO_POINTS = [
    (0.0, (48, 18, 59)),
    (0.15, (70, 107, 227)),
    (0.35, (40, 188, 235)),
    (0.5, (50, 240, 150)),
    (0.65, (195, 230, 45)),
    (0.8, (250, 112, 32)),
    (1.0, (122, 4, 3)),
]

VIRIDIS_POINTS = [
    (0.0, (68, 1, 84)),
    (0.15, (71, 44, 122)),
    (0.30, (59, 81, 139)),
    (0.45, (44, 113, 142)),
    (0.60, (33, 144, 141)),
    (0.75, (94, 201, 98)),
    (1.0, (253, 231, 37)),
]


def interpolated_color(t, points):
    t = max(0.0, min(1.0, float(t)))
    for i in range(len(points) - 1):
        t1, c1 = points[i]
        t2, c2 = points[i + 1]
        if t1 <= t <= t2:
            factor = (t - t1) / (t2 - t1)
            r = int(c1[0] + factor * (c2[0] - c1[0]))
            g = int(c1[1] + factor * (c2[1] - c1[1]))
            b = int(c1[2] + factor * (c2[2] - c1[2]))
            return (r, g, b)
    return points[-1][1]


def turbo_color(t):
    return interpolated_color(t, TURBO_POINTS)


def viridis_color(t):
    return interpolated_color(t, VIRIDIS_POINTS)


def heatmap_color(t, palette_idx):
    if palette_idx == 1:
        return viridis_color(t)
    return turbo_color(t)


def score_to_heatmap_t(score, min_s, max_s, scale_mode):
    diff = max_s - min_s if max_s > min_s else 1.0
    if scale_mode == 0:
        t = (score - min_s) / diff
        return min(1.0, t / 0.75)

    min_s_safe = max(1.0, min_s)
    max_ratio = max_s / min_s_safe
    max_log = math.log(max_ratio) if max_ratio > 1.0 else 1.0
    s_norm = score / max(1.0, min_s)
    val_log = math.log(max(1.0, s_norm))
    return val_log / max_log if max_log > 0 else 0.0


def cool_colormap(t):
    t = max(0.0, min(1.0, float(t)))
    return (int(255 * t), int(255 * (1.0 - t)), 255)


def interpolate_regular_score(world_x, world_y, score_grid, grid_spacing_mm):
    """Interpolate a regular-grid score, falling back to a nearby grid point."""
    grid_x = world_x / grid_spacing_mm
    grid_y = world_y / grid_spacing_mm
    index_x = math.floor(grid_x)
    index_y = math.floor(grid_y)
    fraction_x = grid_x - index_x
    fraction_y = grid_y - index_y

    q00 = score_grid.get((index_x, index_y))
    q10 = score_grid.get((index_x + 1, index_y))
    q01 = score_grid.get((index_x, index_y + 1))
    q11 = score_grid.get((index_x + 1, index_y + 1))
    if all(score is not None for score in (q00, q10, q01, q11)):
        return (
            q00 * (1.0 - fraction_x) * (1.0 - fraction_y)
            + q10 * fraction_x * (1.0 - fraction_y)
            + q01 * (1.0 - fraction_x) * fraction_y
            + q11 * fraction_x * fraction_y
        )

    candidates = []
    for candidate_x, candidate_y, score in (
        (index_x, index_y, q00),
        (index_x + 1, index_y, q10),
        (index_x, index_y + 1, q01),
        (index_x + 1, index_y + 1, q11),
    ):
        if score is not None:
            distance_squared = (world_x - candidate_x * grid_spacing_mm) ** 2 + (world_y - candidate_y * grid_spacing_mm) ** 2
            candidates.append((distance_squared, score))
    if not candidates:
        return None
    distance_squared, score = min(candidates, key=lambda item: item[0])
    return score if distance_squared <= (grid_spacing_mm * 1.45) ** 2 else None


def edge_weight_log_scale(edge_weights, block_weight):
    finite_values = [value for value in edge_weights.values() if value < block_weight]
    max_ratio = max(finite_values) if finite_values else 1.0
    return max_ratio, math.log1p(max(max_ratio, 1.0))
