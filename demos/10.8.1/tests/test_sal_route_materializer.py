from types import SimpleNamespace

from mep_routing.installations.sal import SalRouteMaterializer


class SpatialIndex:
    def query(self, point):
        return 0.0, int(point[0])


def _materializer(add_shaft_entry_segments=None):
    return SalRouteMaterializer(
        env_nodes=((0, 0), (10, 0), (20, 0)),
        spatial_index=SpatialIndex(),
        route_plan=SimpleNamespace(shaft_route="Shaft"),
        add_shaft_entry_segments=add_shaft_entry_segments,
    )


def test_materializer_resolves_source_nodes_from_live_spatial_index():
    assert _materializer().source_start_nodes((2.5, 0.0)) == [2]


def test_materializer_adds_shaft_geometry_and_machine_pin_stub():
    materializer = _materializer(lambda segments, _first: segments.insert(0, ((-10, 0), (0, 0))))

    segments = materializer.route_segments(
        "Shaft",
        [0, 1],
        pin_name="left_mid",
        global_pins={"left_mid": (15, 0)},
    )

    assert segments == [((-10, 0), (0, 0)), ((0, 0), (10, 0)), ((10, 0), (15, 0))]


def test_materializer_builds_all_routes_with_one_route_plan():
    routes, total_nodes = _materializer().build_routes(
        ["Shaft", "Bathroom"],
        {"Shaft": [0, 1], "Bathroom": [1, 2]},
        {"Shaft": {"pin": None}, "Bathroom": {"pin": None}},
        {},
    )

    assert [name for name, _segments in routes] == ["Shaft", "Bathroom"]
    assert total_nodes == 4
