import numpy as np
from shapely.geometry import box

from mep_routing.routing.shaft_entries import (
    select_shaft_entry_nodes,
    shaft_entry_geometry,
    shaft_entry_segments,
)


def test_generic_shaft_entry_prefers_nearest_radial_node_and_builds_segments():
    shaft = box(0, 0, 100, 100)
    nodes = np.array([(150, 50), (250, 50)], dtype=np.float32)

    candidates, chosen, geometries = select_shaft_entry_nodes(
        nodes, shaft, search_radius_mm=500, grid_spacing_mm=200, max_candidates=2,
    )

    assert candidates == [0, 1]
    assert chosen == 0
    assert geometries[0]["entry"] == (100.0, 50.0)
    assert shaft_entry_segments(geometries[0]) == [((50.0, 50.0), (100.0, 50.0)), ((100.0, 50.0), (150.0, 50.0))]


def test_core_entry_spec_rejects_nodes_behind_its_outward_normal():
    shaft = box(0, 0, 100, 100)
    nodes = np.array([(50, 50), (150, 50), (100, 150)], dtype=np.float32)
    spec = {"centroid": (50, 50), "entry": (100, 50), "normal": (1, 0), "exit_wall": None}

    candidates, chosen, geometries = select_shaft_entry_nodes(
        nodes, shaft, search_radius_mm=500, grid_spacing_mm=200, max_candidates=2, core_entry_specs=[spec],
    )

    assert candidates == [1, 2]
    assert chosen == 1
    assert geometries[1]["source"] == "routing_core"
    assert shaft_entry_geometry(shaft, nodes[1])["orthogonality_error"] == 0.0
