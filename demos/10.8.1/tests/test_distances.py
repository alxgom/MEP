import numpy as np

from vent_router.geometry import (
    edge_parallel_segment_min_distances,
    edge_segment_min_distances,
    point_segment_min_distances,
)


def test_point_segment_min_distances_handles_projection_and_endpoint_cases():
    points = np.array([[5.0, 3.0], [15.0, 0.0]])
    segments = np.array([[0.0, 0.0, 10.0, 0.0]])

    distances = point_segment_min_distances(points, segments)

    np.testing.assert_allclose(distances, [3.0, 5.0])


def test_point_segment_min_distances_returns_inf_without_segments():
    distances = point_segment_min_distances(np.array([[1.0, 2.0]]), np.empty((0, 4)))

    assert np.isinf(distances[0])


def test_edge_segment_min_distances_samples_edge_points():
    edges = np.array([[0.0, 5.0, 10.0, 5.0]])
    segments = np.array([[0.0, 0.0, 10.0, 0.0]])

    distances = edge_segment_min_distances(edges, segments, sample_count=3)

    np.testing.assert_allclose(distances, [5.0])


def test_edge_parallel_segment_min_distances_only_counts_parallel_overlap():
    edges = np.array([
        [0.0, 5.0, 10.0, 5.0],
        [20.0, 5.0, 30.0, 5.0],
        [5.0, 0.0, 5.0, 10.0],
    ])
    segments = np.array([[0.0, 0.0, 10.0, 0.0]])

    distances = edge_parallel_segment_min_distances(edges, segments)

    assert distances[0] == 5.0
    assert np.isinf(distances[1])
    assert np.isinf(distances[2])

