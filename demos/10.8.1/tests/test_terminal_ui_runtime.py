from types import SimpleNamespace

from shapely.geometry import box

import mep_routing.ui.terminal_runtime as terminal_ui_module
from mep_routing.ui.terminal_runtime import TerminalUiRuntime, TerminalUiStyle


class Terminals:
    preferred_areas = []
    preferred_points_by_room = {}

    def room_polygon(self, name):
        return {"Kitchen": box(0, 0, 100, 100), "Small": box(0, 0, 20, 20)}.get(name)

    def valid_region(self, name):
        return self.room_polygon(name)

    def boundary_segments(self, name):
        return ()

    def candidate_nodes(self, name):
        return [0]

    def route_start_nodes(self, name, *, use_nearest_terminal):
        return [1] if use_nearest_terminal else [0]

    def find_candidate_at(self, point):
        return "Kitchen", 0

    def apply_point_preference(self, point, *, remove):
        return True, "Kitchen"

    def apply_area_preference(self, start, end, *, remove):
        return True, "Kitchen"


def runtime():
    viewport = SimpleNamespace(scale=1.0, to_screen=lambda x, y: (x, y))
    canvas = SimpleNamespace(viewport=viewport)
    style = TerminalUiStyle({}, (1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), 12, 30, 20, 15)
    result = TerminalUiRuntime(canvas, style)
    env = SimpleNamespace(nodes=[(10, 10)], adj={0: []})
    dwelling = SimpleNamespace(
        terminals=Terminals(), workspace=SimpleNamespace(env=env),
        terminal_points={"Kitchen": (10, 10), "Small": (10, 10)},
        routing_region=box(-10, -10, 200, 200),
    )
    result.bind(dwelling)
    return result


def test_bound_runtime_delegates_terminal_interactions():
    ui = runtime()
    assert ui.route_start_nodes("Kitchen") == [0]
    assert ui.route_start_nodes("Kitchen", use_nearest_terminal=True) == [1]
    assert ui.find_candidate((10, 10)) == ("Kitchen", 0)
    assert ui.apply_point_preference((10, 10)) == (True, "Kitchen")
    assert ui.apply_area_preference((0, 0), (20, 20)) == (True, "Kitchen")


def test_room_hit_prefers_smallest_polygon_and_handles_shaft():
    ui = runtime()
    assert ui.find_room_route((10, 10), ["Kitchen", "Small"]) == "Small"
    assert ui.find_room_route((150, 150), ["Shaft"], box(140, 140, 160, 160)) == "Shaft"


def test_validity_cache_recomputes_only_after_invalidate(monkeypatch):
    calls = []
    monkeypatch.setattr(
        terminal_ui_module, "terminal_validity_entries",
        lambda *args: (calls.append(args) or ([(10, 10, 0, True)], {})),
    )
    ui = runtime()
    assert ui.validity_entries()[0]
    assert ui.validity_entries()[0]
    assert len(calls) == 1
    ui.invalidate()
    ui.validity_entries()
    assert len(calls) == 2
