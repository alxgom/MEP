from types import SimpleNamespace

from shapely.geometry import box

from mep_routing.app import CommonGraphConfig, TerminalRuntimeConfig
from mep_routing.data_sources.scenario import PreparedDwelling
from mep_routing.installations.cli import (
    CLI_INSTALLATION,
    CliInteractiveRuntime,
    cli_machine_pins,
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
    assert result.status.startswith("Cli supply tree solved")
    assert result.workspace.env is not None
