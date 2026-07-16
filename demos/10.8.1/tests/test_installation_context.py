from types import SimpleNamespace

from mep_routing.app import installation_context_obstacles


def test_installation_context_projects_machine_and_route_geometry():
    result = SimpleNamespace(
        machine_pins={
            "c_tl": (0, 0), "c_tr": (100, 0),
            "c_br": (100, 50), "c_bl": (0, 50),
        },
        routes=(("Supply", [((200, 200), (500, 200))]),),
    )
    obstacles = installation_context_obstacles((result,), route_clearance_mm=25)
    assert len(obstacles) == 2
    assert obstacles[0].area == 5000
    assert obstacles[1].area == 15000
