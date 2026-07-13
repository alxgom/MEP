from __future__ import annotations

import numpy as np


def point_segment_min_distances(points, segments, chunk_size=128):
    points = np.asarray(points, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.float64)
    if len(points) == 0:
        return np.empty((0,), dtype=np.float64)
    if len(segments) == 0:
        return np.full((len(points),), np.inf, dtype=np.float64)

    out = np.full((len(points),), np.inf, dtype=np.float64)
    ax = segments[:, 0]
    ay = segments[:, 1]
    bx = segments[:, 2]
    by = segments[:, 3]
    vx = bx - ax
    vy = by - ay
    denom = np.maximum(vx * vx + vy * vy, 1e-9)

    for start in range(0, len(points), chunk_size):
        pts = points[start:start + chunk_size]
        px = pts[:, 0:1]
        py = pts[:, 1:2]
        t = ((px - ax) * vx + (py - ay) * vy) / denom
        t = np.clip(t, 0.0, 1.0)
        cx = ax + t * vx
        cy = ay + t * vy
        out[start:start + chunk_size] = np.sqrt(np.min((px - cx) ** 2 + (py - cy) ** 2, axis=1))
    return out


def edge_segment_min_distances(edge_coords, segments, sample_count=5, chunk_size=128):
    edge_coords = np.asarray(edge_coords, dtype=np.float64)
    if len(edge_coords) == 0:
        return np.empty((0,), dtype=np.float64)
    if len(segments) == 0:
        return np.full((len(edge_coords),), np.inf, dtype=np.float64)

    samples_t = np.linspace(0.0, 1.0, int(sample_count), dtype=np.float64)
    out = np.full((len(edge_coords),), np.inf, dtype=np.float64)
    for start in range(0, len(edge_coords), chunk_size):
        coords = edge_coords[start:start + chunk_size]
        xs = coords[:, 0:1] + (coords[:, 2:3] - coords[:, 0:1]) * samples_t
        ys = coords[:, 1:2] + (coords[:, 3:4] - coords[:, 1:2]) * samples_t
        sample_points = np.column_stack([xs.ravel(), ys.ravel()])
        sample_distances = point_segment_min_distances(sample_points, segments)
        out[start:start + chunk_size] = sample_distances.reshape(len(coords), len(samples_t)).min(axis=1)
    return out


def edge_parallel_segment_min_distances(edge_coords, segments, eps=1e-7, chunk_size=512):
    edge_coords = np.asarray(edge_coords, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.float64)
    if len(edge_coords) == 0:
        return np.empty((0,), dtype=np.float64)
    if len(segments) == 0:
        return np.full((len(edge_coords),), np.inf, dtype=np.float64)

    sx1 = np.minimum(segments[:, 0], segments[:, 2])
    sx2 = np.maximum(segments[:, 0], segments[:, 2])
    sy1 = np.minimum(segments[:, 1], segments[:, 3])
    sy2 = np.maximum(segments[:, 1], segments[:, 3])
    seg_h = np.abs(segments[:, 1] - segments[:, 3]) <= eps
    seg_v = np.abs(segments[:, 0] - segments[:, 2]) <= eps

    out = np.full((len(edge_coords),), np.inf, dtype=np.float64)
    for start in range(0, len(edge_coords), chunk_size):
        coords = edge_coords[start:start + chunk_size]
        ex1 = np.minimum(coords[:, 0], coords[:, 2])[:, None]
        ex2 = np.maximum(coords[:, 0], coords[:, 2])[:, None]
        ey1 = np.minimum(coords[:, 1], coords[:, 3])[:, None]
        ey2 = np.maximum(coords[:, 1], coords[:, 3])[:, None]
        edge_h = (np.abs(coords[:, 1] - coords[:, 3]) <= eps)[:, None]
        edge_v = (np.abs(coords[:, 0] - coords[:, 2]) <= eps)[:, None]

        h_overlap = (np.minimum(ex2, sx2) - np.maximum(ex1, sx1)) > eps
        h_dist = np.abs(coords[:, 1:2] - segments[:, 1])
        h_mask = edge_h & seg_h & h_overlap

        v_overlap = (np.minimum(ey2, sy2) - np.maximum(ey1, sy1)) > eps
        v_dist = np.abs(coords[:, 0:1] - segments[:, 0])
        v_mask = edge_v & seg_v & v_overlap

        d = np.full((len(coords), len(segments)), np.inf, dtype=np.float64)
        d[h_mask] = h_dist[h_mask]
        d[v_mask] = v_dist[v_mask]
        out[start:start + chunk_size] = np.min(d, axis=1)
    return out

