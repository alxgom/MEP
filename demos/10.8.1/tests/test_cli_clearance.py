from types import SimpleNamespace

import numpy as np
from shapely.geometry import box

from mep_routing.graphs import create_runtime_graph
from mep_routing.installations.cli import CLI_INSTALLATION
from mep_routing.installations.cli.clearance import (
    CLI_MINIMUM_CENTERLINE_CLEARANCE_MM,
    CLI_OBSTACLE_WEIGHT,
    apply_cli_edge_weights,
    cli_minimum_clearance_weights,
)


def test_cli_minimum_clearance_is_half_the_minimum_duct_width():
    assert CLI_MINIMUM_CENTERLINE_CLEARANCE_MM == 75


def test_cli_clearance_weights_parallel_wall_and_machine_edges():
    nodes = np.array([
        [100, 50], [900, 50],
        [100, 150], [900, 150],
        [450, 500], [550, 500],
    ], dtype=np.int32)
    edges = [
        (0, 1, 800.0, "E"),
        (2, 3, 800.0, "E"),
        (4, 5, 100.0, "E"),
    ]
    runtime = create_runtime_graph(nodes, edges)
    prepared = SimpleNamespace(
        routing_region_base=box(0, 0, 1000, 1000),
        rooms=[], walls=[], wall_polygons=[],
    )
    machine = SimpleNamespace(body_size_mm=(100, 100, 100))

    weights = cli_minimum_clearance_weights(
        prepared, machine, (500, 500), 0, runtime,
    )

    assert weights[(0, 1)] == CLI_OBSTACLE_WEIGHT
    assert (2, 3) not in weights
    assert weights[(4, 5)] == CLI_OBSTACLE_WEIGHT
    weighted = apply_cli_edge_weights(runtime.adjacency, weights)
    assert weighted[0][0][1] == CLI_OBSTACLE_WEIGHT
    assert weighted[2][0][1] == 800.0
