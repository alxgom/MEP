from types import SimpleNamespace

from shapely.geometry import Polygon

from mep_routing.data_sources import build_synthetic_dwelling


class FakeRoom:
    def __init__(self, polygon, name):
        self.polygon = polygon
        self.name = name


class FakeLayoutProvider:
    Room = FakeRoom

    @staticmethod
    def generate_layout(**_kwargs):
        return [
            FakeRoom(Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]), "Bathroom"),
            FakeRoom(Polygon([(5, 0), (10, 0), (10, 5), (5, 5)]), "Kitchen"),
        ]

    @staticmethod
    def generate_mep_shafts(_rooms):
        return [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]

    @staticmethod
    def generate_structural_grid(_boundary, spacing):
        assert spacing == 4.0
        return []

    @staticmethod
    def find_door_openings(_rooms):
        return [{"d1": (5, 1), "d2": (5, 2), "swing_dir": "left", "width": 1.0}]

    @staticmethod
    def find_entrance_door(_rooms, _boundary):
        return {"d1": (10, 1), "d2": (10, 2), "swing_dir": "right", "width": 1.0}


def test_build_synthetic_dwelling_normalizes_geometry_and_returns_routing_inputs():
    scenario = build_synthetic_dwelling(
        FakeLayoutProvider,
        FakeRoom,
        lambda polygon: (round(polygon.centroid.x), round(polygon.centroid.y)),
    )

    assert [room.name for room in scenario.rooms] == ["Bathroom", "Kitchen"]
    assert scenario.rooms[0].has_cover is True
    assert scenario.rooms[0].polygon.bounds == (0.0, 0.0, 5000.0, 5000.0)
    assert scenario.doors[-1]["is_entrance"] is True
    assert scenario.terminals == {"Bathroom": (2500, 2500), "Kitchen": (7500, 2500)}
    assert scenario.machine_position == (2500, 2500)
    assert scenario.routing_region_base.contains(scenario.rooms[1].polygon.centroid)
