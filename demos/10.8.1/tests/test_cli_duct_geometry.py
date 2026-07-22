from types import SimpleNamespace

import pytest
from shapely.geometry import box

from mep_routing.installations.cli import (
    CliDuctAlignmentConstraints,
    build_aligned_duct_polygon_projection,
    build_core_duct_polygon_projection,
)
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


def test_aligned_projection_does_not_add_decisions_to_unchanged_straight_run():
    sizing = SimpleNamespace(airflow_m3h=500.0)
    segments = (
        CliDuctSegment((0, 0), (1000, 0), 300, 150, "machine_stub", "Supply", sizing),
        CliDuctSegment((1000, 0), (2000, 0), 300, 150, "tree", "Supply", sizing),
    )

    projection = build_aligned_duct_polygon_projection(segments)

    assert projection.alignment_states == (0, 0)
    assert projection.alignment_offsets_mm == (0.0, 0.0)
    assert projection.event_count == 0
    assert projection.junction_polygons == ()


def test_aligned_projection_keeps_high_flow_tee_run_away_from_secondary_branch():
    segments = (
        CliDuctSegment(
            (-1000, 0), (0, 0), 400, 150, "machine_stub", "Supply",
            SimpleNamespace(airflow_m3h=1000.0),
        ),
        CliDuctSegment(
            (1000, 0), (0, 0), 300, 150, "tree", "Main",
            SimpleNamespace(airflow_m3h=800.0),
        ),
        CliDuctSegment(
            (0, 0), (0, 1000), 150, 150, "tree", "Branch",
            SimpleNamespace(airflow_m3h=200.0),
        ),
    )

    projection = build_aligned_duct_polygon_projection(segments)

    assert projection.alignment_states == (0, -1, 0)
    assert projection.alignment_offsets_mm == (0.0, -50.0, 0.0)
    assert projection.event_count == 1
    assert projection.conducto_polygons[1].bounds == pytest.approx((0, -200, 1000, 100))


def test_aligned_projection_propagates_alignment_through_downstream_reduction():
    segments = (
        CliDuctSegment(
            (-1000, 0), (0, 0), 400, 150, "machine_stub", "Supply",
            SimpleNamespace(airflow_m3h=1000.0),
        ),
        CliDuctSegment(
            (0, 0), (1000, 0), 300, 150, "tree", "Main",
            SimpleNamespace(airflow_m3h=800.0),
        ),
        CliDuctSegment(
            (0, 0), (0, 1000), 150, 150, "tree", "Branch",
            SimpleNamespace(airflow_m3h=200.0),
        ),
        CliDuctSegment(
            (1000, 0), (2000, 0), 200, 150, "tree", "Main",
            SimpleNamespace(airflow_m3h=500.0),
        ),
    )

    projection = build_aligned_duct_polygon_projection(segments)

    assert projection.alignment_states == (0, -1, 0, -1)
    assert projection.alignment_offsets_mm == (0.0, -50.0, 0.0, -100.0)
    assert projection.event_count == 2
    assert projection.conducto_polygons[3].bounds == pytest.approx((1000, -200, 2000, 0))


def test_aligned_projection_keeps_machine_side_reduction_centered():
    segments = (
        CliDuctSegment((0, 0), (200, 0), 250, 150, "machine_stub", "Supply"),
        CliDuctSegment((200, 0), (500, 0), 400, 150, "machine_branch", "Supply"),
        CliDuctSegment((500, 0), (1000, 0), 400, 150, "tree", "Supply"),
    )

    projection = build_aligned_duct_polygon_projection(segments)

    assert projection.alignment_states == (0, 0, 0)
    assert projection.alignment_offsets_mm == (0.0, 0.0, 0.0)
    assert projection.conducto_polygons[1].bounds == pytest.approx((200, -200, 500, 200))


def test_aligned_projection_moves_first_reduction_only_when_center_hits_obstacle():
    segments = (
        CliDuctSegment((0, 0), (200, 0), 250, 150, "machine_stub", "Supply"),
        CliDuctSegment((200, 0), (500, 0), 400, 150, "machine_branch", "Supply"),
        CliDuctSegment((500, 0), (1000, 0), 400, 150, "tree", "Supply"),
    )
    constraints = CliDuctAlignmentConstraints(
        obstacles=(box(200, -260, 1000, -130),),
    )

    projection = build_aligned_duct_polygon_projection(segments, constraints)

    assert projection.alignment_states == (0, -1, -1)
    assert projection.alignment_offsets_mm == (0.0, 75.0, 75.0)
    assert projection.conducto_polygons[1].bounds == pytest.approx((200, -125, 500, 275))


def test_aligned_projection_handles_disconnected_endpoint_adapters():
    segments = (
        CliDuctSegment((0, 0), (200, 0), 250, 150, "machine_stub", "Supply"),
        CliDuctSegment((300, 0), (1000, 0), 400, 150, "tree", "Supply"),
        CliDuctSegment((1000, 0), (1000, 800), 200, 150, "tree", "Branch"),
    )

    projection = build_aligned_duct_polygon_projection(segments)

    assert len(projection.conducto_polygons) == 3
    assert len(projection.alignment_offsets_mm) == 3


def test_cli_runtime_reuses_aligned_projection_for_same_result():
    from mep_routing.installations.cli.runtime import CliInteractiveRuntime

    runtime = object.__new__(CliInteractiveRuntime)
    runtime._aligned_polygon_result = None
    runtime._aligned_polygon_projection = None
    result = SimpleNamespace(supply_segments=(
        CliDuctSegment((0, 0), (1000, 0), 300, 150, "tree", "Supply Air Tree"),
    ))

    first = runtime.aligned_polygon_projection(result)
    second = runtime.aligned_polygon_projection(result)

    assert second is first


def test_cli_runtime_supplies_static_obstacles_to_alignment_projection():
    from mep_routing.installations.cli.runtime import CliInteractiveRuntime

    runtime = object.__new__(CliInteractiveRuntime)
    runtime._aligned_polygon_result = None
    runtime._aligned_polygon_projection = None
    result = SimpleNamespace(
        supply_segments=(
            CliDuctSegment((0, 0), (200, 0), 250, 150, "machine_stub", "Supply"),
            CliDuctSegment((200, 0), (1000, 0), 400, 150, "machine_branch", "Supply"),
        ),
        workspace=SimpleNamespace(lifecycle=SimpleNamespace(
            routing_region=box(-500, -500, 1500, 500),
            wall_polygons=(box(200, -260, 1000, -130),),
            columns=(), shafts=(),
        )),
        context_obstacles=(),
        machine_pins={},
    )

    projection = runtime.aligned_polygon_projection(result)

    assert projection.alignment_states == (0, -1)
    assert projection.alignment_offsets_mm == (0.0, 75.0)
