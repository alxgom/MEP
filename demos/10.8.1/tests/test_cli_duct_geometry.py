from types import SimpleNamespace

import pytest

from mep_routing.installations.cli import build_core_duct_polygon_projection
from mep_routing.installations.cli.materializer import CliDuctSegment


def test_core_projection_builds_conducto_rectangles_at_segment_width():
    segment = CliDuctSegment((0, 0), (1000, 0), 300, 150, "tree", "Supply Air Tree")

    projection = build_core_duct_polygon_projection((segment,))

    assert len(projection.conducto_polygons) == 1
    assert projection.conducto_polygons[0].area == pytest.approx(300_000)
    assert projection.union_polygons == ()


def test_core_projection_builds_max_width_square_at_shared_union():
    segments = (
        CliDuctSegment((0, 0), (1000, 0), 300, 150, "tree", "Supply Air Tree"),
        CliDuctSegment((1000, 0), (1000, 800), 200, 150, "tree", "Supply Air Tree"),
    )

    projection = build_core_duct_polygon_projection(segments)

    assert len(projection.union_polygons) == 1
    assert projection.union_polygons[0].area == pytest.approx(300 * 300)


def test_cli_runtime_reuses_projection_for_same_result():
    from mep_routing.installations.cli.runtime import CliInteractiveRuntime

    runtime = object.__new__(CliInteractiveRuntime)
    runtime._core_polygon_result = None
    runtime._core_polygon_projection = None
    result = SimpleNamespace(supply_segments=(
        CliDuctSegment((0, 0), (1000, 0), 300, 150, "tree", "Supply Air Tree"),
    ))

    first = runtime.core_polygon_projection(result)
    second = runtime.core_polygon_projection(result)

    assert second is first
