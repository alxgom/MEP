from types import SimpleNamespace

from shapely.geometry import Point, box

from mep_routing.app import CommonGraphConfig, TerminalRuntimeConfig
from mep_routing.data_sources.scenario import PreparedDwelling
from mep_routing.installations.cli import (
    CLI_INSTALLATION,
    CliInteractiveRuntime,
    CliPlacementRuntime,
    cli_machine_pins,
    prepare_cli_grilles,
    prepare_cli_supply_terminals,
)


def _room(name, bounds, *, covered=True):
    return SimpleNamespace(name=name, category=name, type_name="", polygon=box(*bounds), has_cover=covered)


def _prepared():
    region = box(0, 0, 6000, 5000)
    rooms = [
        _room("Living", (200, 200, 2800, 2400)),
        _room("Bedroom", (3200, 200, 5800, 2400)),
        _room("Kitchen", (200, 2800, 2800, 4800)),
    ]
    return PreparedDwelling(
        rooms=rooms, columns=[], shafts=[], covers=[region], doors=[], walls=[], wall_polygons=[],
        routing_region_base=region, shaft_extraction=None, terminals={}, wet_room_names=[],
        wet_room_outer_accents=[], machine_position=(3000, 3500), shaft_core_entry_specs=[],
        label="cli-test", summary={},
    )


def test_cli_terminals_include_occupied_rooms_and_exclude_service_rooms():
    prepared = _prepared()
    terminals = prepare_cli_supply_terminals(prepared.rooms, prepared.routing_region_base)
    assert set(terminals) == {"Living Supply", "Bedroom Supply"}


def test_cli_machine_geometry_changes_with_selected_catalog_size():
    small = CLI_INSTALLATION.machine_catalog.value("pead-m35ja")
    large = CLI_INSTALLATION.machine_catalog.value("pead-m140ja")
    small_pins = cli_machine_pins(small, (3000, 3500), 0)
    large_pins = cli_machine_pins(large, (3000, 3500), 0)
    assert small_pins["c_tr"][0] - small_pins["c_tl"][0] == 1028
    assert large_pins["c_tr"][0] - large_pins["c_tl"][0] == 1728


def test_cli_runtime_builds_core_simple_supply_tree():
    runtime = CliInteractiveRuntime(
        CLI_INSTALLATION,
        CommonGraphConfig(200, 600, 100, 0, 200),
        TerminalRuntimeConfig(100, 100, 300),
    )
    result = runtime.solve(
        _prepared(), CLI_INSTALLATION.default_machine, (3000, 3500), 0,
    )
    assert result.routes[0][0] == "Supply Air Tree"
    assert result.routes[0][1]
    assert result.total_nodes > 0
    assert result.status.startswith("Cli supply/return solved")
    assert result.workspace.env is not None
    assert [name for name, _segments in result.routes] == ["Supply Air Tree", "Return Air Tree"]
    assert {segment.kind for segment in result.supply_segments} >= {"tree", "machine_stub"}
    assert result.return_segments
    assert all(segment.width_mm > 0 and segment.height_mm > 0 for segment in result.supply_segments)


def test_cli_runtime_tracks_ordered_context_obstacles():
    runtime = CliInteractiveRuntime(
        CLI_INSTALLATION,
        CommonGraphConfig(200, 600, 100, 0, 200),
        TerminalRuntimeConfig(100, 100, 300),
    )
    result = runtime.solve(
        _prepared(), CLI_INSTALLATION.default_machine, (3000, 3500), 0,
        context_obstacles=(box(50, 50, 100, 100),),
    )
    assert result.context_obstacle_count == 1
    assert result.routes


def test_cli_grilles_prepare_supply_and_return_per_occupied_room():
    prepared = _prepared()
    plan = prepare_cli_grilles(prepared.rooms, prepared.routing_region_base)
    assert {item.role for item in plan.connections} == {"supply", "return"}
    assert len(plan.supply) == 2
    assert len(plan.returns) == 2
    assert all(item.route_name.endswith(("Supply", "Return")) for item in plan.connections)


def test_cli_placement_uses_selected_machine_footprint():
    prepared = _prepared()
    plan = prepare_cli_grilles(prepared.rooms, prepared.routing_region_base)
    placement = CliPlacementRuntime(spacing_mm=600)
    small = placement.place(prepared, CLI_INSTALLATION.machine_catalog.value("pead-m35ja"), plan)
    large = placement.place(prepared, CLI_INSTALLATION.machine_catalog.value("pead-m140ja"), plan)
    assert small.feasible_candidates >= large.feasible_candidates > 0
    assert prepared.routing_region_base.covers(Point(small.center))
