import numpy as np
from shapely.geometry import box

from mep_routing.routing.terminal_runtime import TerminalRuntime


class NearestIndex:
    def __init__(self, nodes):
        self.nodes = np.asarray(nodes, dtype=np.float32)

    def query(self, point):
        distances = np.hypot(self.nodes[:, 0] - point[0], self.nodes[:, 1] - point[1])
        index = int(np.argmin(distances))
        return float(distances[index]), index


def make_runtime(*, terminals=None, room_polygons=None, routing_region=None):
    nodes = np.asarray(((5, 5), (7, 5), (9, 5), (15, 5)), dtype=np.float32)
    return TerminalRuntime(
        terminals=terminals or {"bath": (5, 5)},
        room_polygons=room_polygons if room_polygons is not None else {"bath": box(0, 0, 10, 10)},
        routing_region=routing_region if routing_region is not None else box(0, 0, 10, 10),
        covers=(),
        walls=(),
        wall_polygons=(),
        regulation_clearance_mm=1.0,
        terminal_buffer_mm=1.0,
        remap_tolerance_mm=2.0,
        nodes=nodes,
        adjacency={0: [(1, 2.0)], 1: [(0, 2.0)], 2: [(1, 2.0)], 3: [(2, 6.0)]},
        nearest_index=NearestIndex(nodes),
    )


def test_runtime_uses_nearest_node_when_room_constraints_are_unavailable():
    runtime = make_runtime(room_polygons={})

    assert runtime.candidate_nodes("bath") == [0]
    assert runtime.route_start_nodes("bath", use_nearest_terminal=True) == [0]


def test_runtime_caches_candidates_until_explicit_graph_invalidation():
    terminals = {"bath": (5, 5)}
    runtime = make_runtime(terminals=terminals)

    assert runtime.candidate_nodes("bath") == [0, 1, 2]
    terminals["bath"] = (9, 5)
    assert runtime.candidate_nodes("bath") == [0, 1, 2]

    runtime.invalidate_candidates()
    assert runtime.candidate_nodes("bath") == [2, 1, 0]


def test_runtime_preferred_nodes_override_default_route_start_nodes():
    runtime = make_runtime()

    changed, room = runtime.apply_point_preference((7.1, 5.0))

    assert (changed, room) == (True, "bath")
    assert runtime.preferred_nodes("bath") == [1]
    assert runtime.route_start_nodes("bath") == [1]


def test_runtime_finds_room_candidates_and_removes_area_preferences():
    runtime = make_runtime()

    assert runtime.find_candidate_at((7.2, 5.0)) == ("bath", 1)
    assert runtime.find_candidate_at((15.0, 5.0)) is None

    added = runtime.apply_area_preference((4.0, 4.0), (8.0, 6.0))
    removed = runtime.apply_area_preference((4.0, 4.0), (8.0, 6.0), remove=True)

    assert added == (True, "bath")
    assert removed == (True, "bath")
    assert runtime.preferred_points_by_room == {}
    assert runtime.preferred_areas == []

    runtime.restore_preferences({"bath": [(7.0, 5.0)]}, [{"room": "bath", "bounds": (6, 4, 8, 6)}])
    assert runtime.preferred_nodes("bath") == [1]
    runtime.clear_preferences()
    assert runtime.preferred_points_by_room == {}
