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
