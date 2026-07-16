from types import SimpleNamespace

from mep_routing.data_sources import build_core_shaft_entry_specs, transform_source_point


def test_transform_source_point_applies_inverse_rotation_and_bounds_origin():
    summary = {
        "routing_frame": {"rotation_degrees": 90},
        "metadata": {"source_rotation_origin": [0, 0], "bounds_origin": [1, -1]},
    }

    assert transform_source_point((0, 2), summary, scale_to_mm=1000) == (1000, 1000)


def test_core_shaft_specs_select_explicit_candidate_and_normalize_entries():
    summary = {
        "routing_frame": {},
        "metadata": {
            "bounds_origin": [0, 0],
            "shaft_candidate_index": 1,
            "shaft_candidates": [
                {"routing": [{"id": "ignored", "centroid": [0, 0], "entry_point": [1, 0]}]},
                {"routing": [{
                    "id": "selected", "centroid": [1, 1], "entry_points": [[2, 1]],
                    "exit_walls": [[[2, 0], [2, 2]]],
                }]},
            ],
        },
    }

    specs = build_core_shaft_entry_specs(
        SimpleNamespace(), summarize=lambda _scenario: summary,
        scale_to_mm=1000, match_tolerance_mm=10,
    )

    assert specs == [{
        "centroid": (1000.0, 1000.0),
        "entry": (2000.0, 1000.0),
        "normal": (1.0, 0.0),
        "exit_wall": ((2000, 0), (2000, 2000)),
        "id": "selected",
    }]
