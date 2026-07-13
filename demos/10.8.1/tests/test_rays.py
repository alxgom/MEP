import numpy as np

from mep_routing.geometry import cast_rays_numpy, ray_ray_intersections_numpy


def test_cast_rays_numpy_casts_to_rectangle_boundaries():
    boundary_segments = np.array([
        [0.0, 0.0, 10.0, 0.0],
        [10.0, 0.0, 10.0, 10.0],
        [10.0, 10.0, 0.0, 10.0],
        [0.0, 10.0, 0.0, 0.0],
    ])
    points = np.array([[5.0, 5.0]])

    horizontal, vertical = cast_rays_numpy(points, boundary_segments)

    assert sorted(horizontal) == [(5.0, 0.0, 5.0), (5.0, 5.0, 10.0)]
    assert sorted(vertical) == [(5.0, 0.0, 5.0), (5.0, 5.0, 10.0)]


def test_ray_ray_intersections_numpy_returns_crossings():
    horizontal = [(5.0, 0.0, 10.0), (8.0, 0.0, 10.0)]
    vertical = [(2.0, 0.0, 10.0), (12.0, 0.0, 10.0)]

    intersections = ray_ray_intersections_numpy(horizontal, vertical)

    assert intersections == [(2.0, 5.0), (2.0, 8.0)]


def test_ray_ray_intersections_numpy_handles_empty_inputs():
    assert ray_ray_intersections_numpy([], [(2.0, 0.0, 10.0)]) == []
    assert ray_ray_intersections_numpy([(5.0, 0.0, 10.0)], []) == []

