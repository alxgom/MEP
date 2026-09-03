import math
from types import SimpleNamespace

from shapely.geometry import Point, box
import pytest

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
from mep_routing.graphs import GraphMode


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


def test_cli_runtime_sizes_only_supply_tree_from_downstream_room_area():
    runtime = CliInteractiveRuntime(
        CLI_INSTALLATION,
        CommonGraphConfig(200, 600, 100, 0, 200),
        TerminalRuntimeConfig(100, 100, 300),
    )

    result = runtime.solve(
        _prepared(), CLI_INSTALLATION.default_machine, (3000, 3500), 0,
    )

    supply_tree = [segment for segment in result.supply_segments if segment.kind == "tree"]
    return_tree = [segment for segment in result.return_segments if segment.kind == "tree"]
    assert supply_tree
    assert all(segment.sizing is not None for segment in supply_tree)
    assert max(segment.sizing.airflow_m3h for segment in supply_tree) == 1260.0
    assert all(segment.height_mm == 150 for segment in supply_tree)
    assert all(segment.sizing is None for segment in return_tree)


def test_cli_supply_materializes_core_sized_endpoint_branches_and_short_stubs():
    runtime = CliInteractiveRuntime(
        CLI_INSTALLATION,
        CommonGraphConfig(200, 600, 100, 0, 200),
        TerminalRuntimeConfig(100, 100, 300),
    )

    result = runtime.solve(
        _prepared(), CLI_INSTALLATION.default_machine, (3000, 3500), 0,
        return_enabled=False,
    )

    machine_stub = next(item for item in result.supply_segments if item.kind == "machine_stub")
    machine_branch = next(item for item in result.supply_segments if item.kind == "machine_branch")
    grille_stubs = [item for item in result.supply_segments if item.kind == "grille_stub"]
    grille_branches = [item for item in result.supply_segments if item.kind == "grille_branch"]

    assert math.dist(machine_stub.start, machine_stub.end) == pytest.approx(200)
    assert machine_stub.height_mm == 150
    assert math.dist(machine_branch.start, machine_branch.end) == pytest.approx(300)
    assert machine_branch.sizing.airflow_m3h == 1260.0
    assert len(grille_stubs) == len(grille_branches) == 2
    assert all(math.dist(item.start, item.end) == pytest.approx(100) for item in grille_stubs)
    assert all(item.sizing is not None for item in grille_branches)
    assert all(item.width_mm < stub.width_mm for item, stub in zip(grille_branches, grille_stubs))
    assert {item.width_mm for item in grille_stubs} == {800}
    assert {item.width_mm for item in result.grille_plan.supply} == {800}
    assert {item.sizing.airflow_m3h for item in result.grille_plan.supply} == {630.0}


def test_cli_runtime_can_disable_return_ventilation():
    runtime = CliInteractiveRuntime(
        CLI_INSTALLATION,
        CommonGraphConfig(200, 600, 100, 0, 200),
        TerminalRuntimeConfig(100, 100, 300),
    )

    result = runtime.solve(
        _prepared(), CLI_INSTALLATION.default_machine, (3000, 3500), 0,
        return_enabled=False,
    )

    assert [name for name, _segments in result.routes] == ["Supply Air Tree"]
    assert result.return_segments == ()
    assert all(not name.endswith(" Return") for name in result.terminals)
    assert result.status.startswith("Cli supply solved")


@pytest.mark.parametrize("backend_index", (3, 4))
def test_cli_directed_wall_candidates_choose_one_sized_supply_grille_per_room(backend_index):
    runtime = CliInteractiveRuntime(
        CLI_INSTALLATION,
        CommonGraphConfig(200, 600, 100, 0, 200),
        TerminalRuntimeConfig(100, 100, 300),
    )

    result = runtime.solve(
        _prepared(), CLI_INSTALLATION.default_machine, (3000, 3500), 0,
        return_enabled=False, solver_backend_index=backend_index, start_mode_index=1,
    )

    assert result.routes
    assert len(result.grille_plan.supply) == 2
    assert all(item.placement_type == "wall_candidate" for item in result.grille_plan.supply)
    assert all(item.grille_point == item.routing_point for item in result.grille_plan.supply)
    assert {item.width_mm for item in result.grille_plan.supply} == {800}
    assert all(item.sizing.airflow_m3h == 630.0 for item in result.grille_plan.supply)


@pytest.mark.parametrize("graph_mode", tuple(GraphMode))
def test_cli_runtime_supports_its_declared_graph_modes(graph_mode):
    runtime = CliInteractiveRuntime(
        CLI_INSTALLATION,
        CommonGraphConfig(200, 600, 100, 0, 200),
        TerminalRuntimeConfig(100, 100, 300),
    )

    result = runtime.solve(
        _prepared(), CLI_INSTALLATION.default_machine, (3000, 3500), 0,
        return_enabled=False, graph_mode=graph_mode,
    )

    assert result.routes
    assert runtime.aligned_polygon_projection(result).conducto_polygons


@pytest.mark.parametrize("backend_index", range(3))
def test_cli_runtime_supports_its_declared_solver_backends(backend_index):
    runtime = CliInteractiveRuntime(
        CLI_INSTALLATION,
        CommonGraphConfig(200, 600, 100, 0, 200),
        TerminalRuntimeConfig(100, 100, 300),
    )

    result = runtime.solve(
        _prepared(), CLI_INSTALLATION.default_machine, (3000, 3500), 0,
        return_enabled=False, solver_backend_index=backend_index,
    )

    assert result.routes


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
    assert {item.width_mm for item in plan.supply} == {1000}
    assert {item.width_mm for item in plan.returns} == {1050}
    assert all(item.placement_type.startswith("common_wall") for item in plan.connections)


def test_cli_grilles_use_horizontal_return_with_partial_false_ceiling():
    room = _room("Living", (0, 0, 4000, 3000))
    false_ceiling = box(2000, 0, 4000, 3000)
    room.space_coverings = [(false_ceiling, False)]
    plan = prepare_cli_grilles((room,), room.polygon)
    assert len(plan.connections) == 2
    assert {item.placement_type for item in plan.connections} == {"return_horizontal"}
    returned = plan.returns[0]
    assert returned.grille_point == returned.routing_point
    assert false_ceiling.covers(Point(returned.grille_point))


def test_cli_placement_uses_selected_machine_footprint():
    prepared = _prepared()
    plan = prepare_cli_grilles(prepared.rooms, prepared.routing_region_base)
    placement = CliPlacementRuntime(spacing_mm=600)
    small = placement.place(prepared, CLI_INSTALLATION.machine_catalog.value("pead-m35ja"), plan)
    large = placement.place(prepared, CLI_INSTALLATION.machine_catalog.value("pead-m140ja"), plan)
    assert small.feasible_candidates >= large.feasible_candidates > 0
    assert prepared.routing_region_base.covers(Point(small.center))
