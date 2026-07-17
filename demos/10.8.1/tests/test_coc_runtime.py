from shapely.geometry import box

from mep_routing.app import CommonGraphConfig, TerminalRuntimeConfig
from mep_routing.data_sources.scenario import PreparedDwelling
from mep_routing.installations.coc import COC_INSTALLATION, CocInteractiveRuntime


def _prepared(coc_input):
    region = box(0, 0, 2000, 1600)
    return PreparedDwelling(
        rooms=[], columns=[], shafts=[], covers=[region], doors=[], walls=[],
        wall_polygons=[], routing_region_base=region, shaft_extraction=None,
        terminals={}, wet_room_names=[], wet_room_outer_accents=[],
        machine_position=(0, 0), shaft_core_entry_specs=[], label="test",
        summary={}, installation_inputs={"coc": coc_input},
    )


def _runtime():
    return CocInteractiveRuntime(
        COC_INSTALLATION,
        CommonGraphConfig(200, 600, 100, 50, 100),
        TerminalRuntimeConfig(100, 100, 100),
    )


def test_coc_routes_exported_connector_to_shaft_in_plan():
    result = _runtime().solve(_prepared({
        "appliance": {
            "selected": True, "source_status": "ready",
            "exhaust_xy": [300, 400], "polygon": box(200, 300, 500, 600),
        },
        "duct": {"diameter_m": 0.125},
        "shaft_entries": ({
            "entry": (1800, 800), "normal": (-1, 0), "centroid": (1900, 800),
        },),
    }))

    assert result.status == "Success: Coc exhaust solved"
    assert result.diameter_mm == 125
    assert result.terminals == {"Coc source": (300.0, 400.0), "Coc shaft": (1800.0, 800.0)}
    assert result.routes[0][0] == "Kitchen Exhaust"
    assert result.routes[0][1][-1] == ((1700.0, 800.0), (1800.0, 800.0))
    assert result.machine_pins["exhaust"] == (300.0, 400.0)


def test_coc_does_not_fabricate_a_missing_exported_source():
    result = _runtime().solve(_prepared({"appliance": None, "shaft_entries": ()}))

    assert not result.routes
    assert result.status == "Blocked: Coc source unavailable"
