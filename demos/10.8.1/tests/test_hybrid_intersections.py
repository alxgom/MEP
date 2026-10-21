"""The numeric orthogonal union must retain GEOS topology and ordering."""

import numpy as np
import pytest
from shapely.geometry import box

from mep_routing.graphs import hybrid


@pytest.mark.parametrize("seed", range(6))
def test_numeric_union_matches_geos_including_indices_overlaps_and_local_links(monkeypatch, seed):
    rng = np.random.default_rng(seed)
    # Fractional coordinates, reversed overlaps, endpoint contacts and crossings.
    segments = [((0, 0), (0, 12)), ((0, 12), (12, 12)),
                ((12, 12), (12, 0)), ((12, 0), (0, 0)),
                ((10, 6), (2, 6)), ((4, 6), (12, 6)),
                ((6, 10), (6, 2)), ((6, 4), (6, 12)),
                ((0, 0), (0, 12))]
    for _ in range(35):
        a, b = rng.integers(0, 49, size=(2, 2)).astype(float) / 4
        axis = int(rng.integers(2))
        b[axis] = a[axis]
        if not np.array_equal(a, b):
            segments.append((a, b))
    nodes = np.asarray(segments, dtype=np.float32).reshape((-1, 2))
    records = [(2 * i, 2 * i + 1) for i in range(len(segments))]
    kwargs = dict(base_nodes=nodes[:8], base_edges=records[:4], added_nodes=nodes[8:],
                  added_edges=[(u - 8, v - 8) for u, v in records[4:]],
                  allowed_region=box(-1, -1, 13, 13), max_link_distance_mm=3)
    numeric = hybrid.join_grids_locally(**kwargs)
    monkeypatch.setattr(hybrid, "_orthogonal_intersections", hybrid._geometry_intersections)
    reference = hybrid.join_grids_locally(**kwargs)
    np.testing.assert_array_equal(numeric[0], reference[0])
    assert numeric[1:] == reference[1:]


@pytest.mark.parametrize("base_end", [(12, 12), (12, 0.00001), (0, 0)])
def test_noncardinal_and_degenerate_inputs_keep_geos_path(monkeypatch, base_end):
    def unexpected_numeric(*args):
        pytest.fail("noncardinal or degenerate segment used the numeric path")

    monkeypatch.setattr(hybrid, "_orthogonal_intersections", unexpected_numeric)
    nodes, edges, _ = hybrid.join_grids_locally(
        np.asarray([(0, 0), base_end], dtype=np.float32), [(0, 1)],
        np.asarray([(3, 0), (3, 12)], dtype=np.float32), [(0, 1)], box(-1, -1, 13, 13))
    assert len(nodes) >= 4 and edges
