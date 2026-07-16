from types import SimpleNamespace

from shapely.geometry import box

from mep_routing.placement import (
    insufficient_machine_clearance_regions,
    is_machine_placement_valid,
)


def _room(*covers):
    return SimpleNamespace(covers=list(covers))


def _cover(tag, height_m, polygon, status="available"):
    return {
        "tag": tag,
        "gross_routing_void_height": height_m,
        "routing_void_status": status,
        "polygon": polygon,
    }


def _pins(x0=0, y0=0, x1=10, y1=10):
    return {"c_tl": (x0, y1), "c_tr": (x1, y1), "c_br": (x1, y0), "c_bl": (x0, y0)}


def test_insufficient_regions_use_height_plus_clearance_and_ignore_unknowns():
    low = _cover("low", 0.20, box(0, 0, 10, 10))
    exact = _cover("exact", 0.22, box(20, 0, 30, 10))
    unknown = _cover("old", None, box(40, 0, 50, 10), "not_exported")
    regions = insufficient_machine_clearance_regions([_room(low, exact, unknown)], 200, 20)
    assert len(regions) == 1
    assert regions[0].equals(box(0, 0, 10, 10))


def test_adequate_overlap_wins_and_boundary_touch_remains_valid():
    low = _cover("low", 0.20, box(0, 0, 20, 20))
    adequate = _cover("ok", 0.30, box(10, 0, 20, 20))
    blocked = insufficient_machine_clearance_regions([_room(low, adequate)], 200, 20)
    assert blocked[0].equals(box(0, 0, 10, 20))
    assert is_machine_placement_valid(15, 5, _pins(10, 0, 20, 10), box(-10, -10, 30, 30), (), (), (), blocked)
    assert not is_machine_placement_valid(5, 5, _pins(1, 1, 9, 9), box(-10, -10, 30, 30), (), (), (), blocked)
