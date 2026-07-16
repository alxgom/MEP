from types import SimpleNamespace

from shapely.geometry import Polygon

from mep_routing.placement import PlacementSession


def _session(graph_builder=lambda **kwargs: None):
    return PlacementSession(
        machine_area=1.0,
        machine_pins=lambda x, y, angle: {
            "body_corners": ((x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1)),
        },
        representative_point=lambda shape: (shape.centroid.x, shape.centroid.y),
        route_room_polygon=lambda name: None,
        local_axis_to_world=lambda angle, axis: axis,
        graph_builder=graph_builder,
        shaft_nodes=lambda env, index: ((), None),
    )


def test_bind_resets_cached_fields_and_tracks_clearance_regions():
    session = _session()
    session.scores, session.fields = {1: 2}, {"shaft": {}}
    region = Polygon(((0, 0), (10, 0), (10, 10), (0, 10)))
    blocked = Polygon(((0, 0), (2, 0), (2, 2), (0, 2)))

    session.bind(SimpleNamespace(), (), {}, (), region, None, (blocked,))

    assert session.scores == {}
    assert session.fields == {}
    assert session.clearance_blocks == (blocked,)
    assert session.placeable_region.area == 96.0


def test_torque_rotation_does_not_rebuild_graph():
    rebuilds = []
    session = _session(lambda **kwargs: rebuilds.append(kwargs))
    session.rotation_scores.update({"selected": "V"})

    angle = session.apply_rotation_mode(0, (5, 5), 90)

    assert angle == 90
    assert session.rotation_scores["selected"] == "Torque"
    assert rebuilds == []

