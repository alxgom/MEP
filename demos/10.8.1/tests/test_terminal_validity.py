import numpy as np
from shapely.geometry import Polygon

from vent_router.routing import terminal_validity_entries


def test_terminal_validity_entries_classifies_room_clearance_and_isolation():
    nodes = np.array(
        [
            [200.0, 200.0],
            [50.0, 200.0],
            [500.0, 500.0],
            [300.0, 300.0],
        ]
    )
    adj = {0: [1], 1: [0], 2: [0], 3: []}
    room = Polygon([(0, 0), (400, 0), (400, 400), (0, 400)])
    boundary_segments = np.array([[0.0, 0.0, 0.0, 400.0]])

    entries, reasons = terminal_validity_entries(
        nodes,
        adj,
        ["Bath"],
        room,
        lambda _name: room,
        lambda _name: room,
        lambda _name: boundary_segments,
        regulation_clearance_mm=100,
        terminal_buffer_mm=150,
    )

    assert entries == [
        (200.0, 200.0, 0, True),
        (50.0, 200.0, 1, False),
        (500.0, 500.0, 2, False),
        (300.0, 300.0, 3, False),
    ]
    assert reasons[0] == ["allowed terminal placement"]
    assert reasons[1] == [
        "inside 100 mm regulation clearance",
        "inside 150 mm terminal buffer",
    ]
    assert reasons[2] == ["outside terminal room"]
    assert reasons[3] == ["isolated graph node"]


def test_terminal_validity_entries_allows_room_nodes_without_boundary_segments():
    nodes = np.array([[200.0, 200.0]])
    adj = {0: [0]}
    room = Polygon([(0, 0), (400, 0), (400, 400), (0, 400)])

    entries, reasons = terminal_validity_entries(
        nodes,
        adj,
        ["Bath"],
        room,
        lambda _name: room,
        lambda _name: room,
        lambda _name: np.empty((0, 4)),
        regulation_clearance_mm=100,
        terminal_buffer_mm=150,
    )

    assert entries == [(200.0, 200.0, 0, True)]
    assert reasons[0] == ["allowed terminal placement"]


def test_terminal_validity_entries_handles_missing_routing_region():
    nodes = np.array([[200.0, 200.0]])

    entries, reasons = terminal_validity_entries(
        nodes,
        {0: [0]},
        ["Bath"],
        None,
        lambda _name: None,
        lambda _name: None,
        lambda _name: np.empty((0, 4)),
        regulation_clearance_mm=100,
        terminal_buffer_mm=150,
    )

    assert entries == [(200.0, 200.0, 0, False)]
    assert reasons[0] == ["outside terminal room"]
