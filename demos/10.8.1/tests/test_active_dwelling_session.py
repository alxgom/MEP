from types import SimpleNamespace

from shapely.geometry import box

from mep_routing.app import ActiveDwellingSession


def test_active_dwelling_session_owns_geometry_and_fresh_runtimes():
    room = SimpleNamespace(name="Bathroom", polygon=box(0, 0, 100, 100))
    prepared = SimpleNamespace(
        rooms=[room], columns=[], shafts=[], covers=[room.polygon], doors=[], walls=[],
        wall_polygons=[], routing_region_base=room.polygon, shaft_extraction=None,
        terminals={"Bathroom": (50, 50)}, wet_room_names=["Bathroom"],
        shaft_core_entry_specs=[],
    )
    lifecycle = object()

    session = ActiveDwellingSession.create(
        prepared,
        lifecycle,
        regulation_clearance_mm=100,
        terminal_buffer_mm=50,
        remap_tolerance_mm=300,
    )

    assert session.prepared is prepared
    assert session.rooms is prepared.rooms
    assert session.routing_region is prepared.routing_region_base
    assert session.terminal_points == {"Bathroom": (50, 50)}
    assert session.workspace.lifecycle is lifecycle
    assert session.terminals.room_polygon("Bathroom") is room.polygon


def test_new_active_dwelling_does_not_reuse_runtime_caches():
    prepared = SimpleNamespace(
        rooms=[], columns=[], shafts=[], covers=[], doors=[], walls=[], wall_polygons=[],
        routing_region_base=box(0, 0, 10, 10), shaft_extraction=None, terminals={},
        wet_room_names=[], shaft_core_entry_specs=[],
    )
    kwargs = dict(regulation_clearance_mm=1, terminal_buffer_mm=1, remap_tolerance_mm=1)

    first = ActiveDwellingSession.create(prepared, object(), **kwargs)
    second = ActiveDwellingSession.create(prepared, object(), **kwargs)

    assert first.workspace is not second.workspace
    assert first.terminals is not second.terminals
