from types import SimpleNamespace

from shapely.geometry import LineString, box

from mep_routing.data_sources.scenario import prepare_real_dwelling, prepare_synthetic_dwelling


def test_prepare_synthetic_dwelling_collects_complete_app_state_and_accents():
    bathroom = SimpleNamespace(name="Bathroom", polygon=box(0, 0, 100, 100))
    kitchen = SimpleNamespace(name="Kitchen", polygon=box(100, 0, 200, 100))
    scenario = SimpleNamespace(
        rooms=[bathroom, kitchen], columns=[], shafts=[], covers=[], doors=[{"id": "door"}], walls=[],
        wall_polygons=[], routing_region_base=box(0, 0, 200, 100), shaft_extraction=None,
        terminals={"Bathroom": (50, 50)}, wet_room_names=["Bathroom"], machine_position=(50, 50),
    )

    prepared = prepare_synthetic_dwelling(scenario)

    assert prepared.label == "synthetic"
    assert prepared.shaft_core_entry_specs == []
    assert prepared.machine_position == (50, 50)
    assert len(prepared.wet_room_outer_accents) == 1


def test_prepare_real_dwelling_uses_explicit_adapters_and_cover_fallback():
    bathroom = SimpleNamespace(name="Bathroom", polygon=box(0, 0, 100, 100), has_cover=True)
    scenario = SimpleNamespace(
        rooms=[bathroom], columns=[box(150, 0, 160, 10)], shafts=[], covers=[],
        shaft_extraction=box(-20, -20, -10, -10), routing_region_base=box(0, 0, 100, 100),
        terminals={"Bathroom": (50, 50)},
    )
    calls = []

    prepared = prepare_real_dwelling(
        scenario,
        wall_thickness_mm=20,
        label="run / dwelling",
        summary={"source": "test"},
        derive_walls=lambda rooms, columns, shafts: calls.append((rooms, columns, shafts)) or [LineString([(0, 0), (100, 0)])],
        build_wall_polygons=lambda walls, columns, shafts, thickness: [box(0, -10, 100, 10)],
        choose_machine_position=lambda terminals, shaft: terminals["Bathroom"],
        build_core_entry_specs=lambda _scenario: [{"id": "shaft-entry"}],
    )

    assert calls
    assert prepared.covers == [bathroom.polygon]
    assert prepared.doors == []
    assert prepared.machine_position == (50, 50)
    assert prepared.shaft_core_entry_specs == [{"id": "shaft-entry"}]
    assert prepared.summary == {"source": "test"}
