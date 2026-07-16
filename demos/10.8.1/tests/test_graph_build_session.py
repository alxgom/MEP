from types import SimpleNamespace

from mep_routing.app import GraphBuildSession


class Workspace:
    grid_available = True
    lifecycle = SimpleNamespace(epsilon_mm=125)

    def build_base_regular(self):
        return SimpleNamespace(nodes=[1, 2])

    def build_selected(self, mode, pins, angle, **kwargs):
        self.selected = mode, pins, angle, kwargs
        return SimpleNamespace(variant=None, elapsed_ms=1.0)

    def apply_machine_obstacle(self, polygon, pins, angle, **kwargs):
        self.dynamic = polygon, pins, angle, kwargs
        return SimpleNamespace(blocked_node_count=2, blocked_edge_count=3)


def test_graph_build_session_commits_explicit_machine_inputs_and_invalidates():
    workspace = Workspace()
    invalidations = []
    session = GraphBuildSession(
        workspace, "terminals",
        lambda: invalidations.append("candidates"),
        lambda: invalidations.append("validity"),
        report=lambda _message: None,
    )

    session.build(0, {"pin": (1, 2)}, 90)

    assert workspace.selected == (
        0, {"pin": (1, 2)}, 90,
        {"shift_hannan_walls": True, "terminal_runtime": "terminals"},
    )
    assert invalidations == ["candidates", "validity"]


def test_graph_build_session_synchronizes_dynamic_graph():
    workspace = Workspace()
    invalidations = []
    session = GraphBuildSession(
        workspace, "terminals", lambda: invalidations.append(True), lambda: None,
        report=lambda _message: None,
    )

    session.apply_machine_obstacle("poly", {"pin": (1, 2)}, 0, clearance_mm=25)

    assert workspace.dynamic == (
        "poly", {"pin": (1, 2)}, 0,
        {"clearance_mm": 25, "terminal_runtime": "terminals"},
    )
    assert invalidations == [True]
