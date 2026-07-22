"""Cli domain and interactive supply-tree runtime."""

from .catalog import CLI_MACHINE_CATALOG, CLI_PEAD_MACHINES, CliConnector, CliMachine
from .definition import CLI_INSTALLATION, CliInstallationDefinition
from .duct_geometry import CliDuctPolygonProjection, build_core_duct_polygon_projection
from .aligned_duct_geometry import (
    CliAlignedDuctProjection, CliDuctAlignmentConstraints,
    build_aligned_duct_polygon_projection,
)
from .geometry import cli_air_access_spec, cli_air_access_specs, cli_machine_pins, cli_supply_access_specs
from .graph_inputs import build_cli_simple_graph_inputs
from .grilles import CliGrilleConnection, CliGrillePlacementConfig, CliGrillePlan, prepare_cli_grilles
from .placement import CliPlacementResult, CliPlacementRuntime
from .route_analysis import CliRouteAnalysis
from .materializer import CliDuctSegment, materialize_cli_tree
from .runtime import CliInteractiveRuntime, CliRoutingResult
from .solver_backends import solve_cli_tree, solve_line_graph_mst
from .sizing import (
    CliDuctSize, CliDuctSizingPolicy, CliGrilleSize, CliGrilleSizingPolicy,
    CliSizingContext, CoreConstantFrictionSizing, CoreSupplyGrilleSizing,
    size_tree_edges,
)
from .terminals import is_cli_supply_room, prepare_cli_supply_terminals

__all__ = [
    "CLI_INSTALLATION", "CLI_MACHINE_CATALOG", "CLI_PEAD_MACHINES",
    "CliConnector", "CliInstallationDefinition", "CliInteractiveRuntime", "CliMachine",
    "CliGrilleConnection", "CliGrillePlacementConfig", "CliGrillePlan", "CliPlacementResult", "CliPlacementRuntime", "CliRouteAnalysis",
    "CliDuctPolygonProjection", "CliDuctSegment", "CliDuctSize", "CliDuctSizingPolicy", "CliSizingContext",
    "CoreConstantFrictionSizing", "CoreSupplyGrilleSizing",
    "CliGrilleSize", "CliGrilleSizingPolicy",
    "CliRoutingResult", "CliAlignedDuctProjection", "CliDuctAlignmentConstraints", "build_aligned_duct_polygon_projection", "build_cli_simple_graph_inputs", "build_core_duct_polygon_projection", "cli_machine_pins",
    "cli_air_access_spec", "cli_air_access_specs", "materialize_cli_tree",
    "cli_supply_access_specs", "is_cli_supply_room", "prepare_cli_supply_terminals",
    "prepare_cli_grilles", "size_tree_edges", "solve_cli_tree", "solve_line_graph_mst",
]
