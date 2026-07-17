"""Declared Cli capabilities consumed by its interactive runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mep_routing.domain import Catalog
from mep_routing.graphs import GraphMode
from mep_routing.routing import SteinerTreeResult, solve_steiner_tree

from .catalog import CLI_MACHINE_CATALOG, CliMachine


CLI_GRAPH_MODES = tuple(GraphMode)
CLI_GRAPH_LABELS = (
    "Regular 200mm Grid",
    "Hannan Grid (numpy)",
    "Epsilon Grid (core-like numpy)",
    "Routing-Core Port: Simple Grid",
)
CLI_STRATEGY_LABELS = ("Multi-terminal Steiner tree",)
CLI_SEARCH_BACKEND_LABELS = ("Kou metric closure",)
CLI_HEURISTIC_MODE_LABELS = ("Shortest-path distance",)
CLI_START_MODE_LABELS = ("Supply/return grilles",)


@dataclass(frozen=True)
class CliInstallationDefinition:
    """Cli catalog, graph, and supply-tree solver capabilities."""

    machine_catalog: Catalog[CliMachine]
    supply_graph_mode: GraphMode
    supply_tree_solver: Callable[..., SteinerTreeResult | None]
    graph_modes: tuple[GraphMode, ...] = CLI_GRAPH_MODES
    graph_labels: tuple[str, ...] = CLI_GRAPH_LABELS
    strategy_labels: tuple[str, ...] = CLI_STRATEGY_LABELS
    search_backend_labels: tuple[str, ...] = CLI_SEARCH_BACKEND_LABELS
    heuristic_mode_labels: tuple[str, ...] = CLI_HEURISTIC_MODE_LABELS
    start_mode_labels: tuple[str, ...] = CLI_START_MODE_LABELS
    interactive_runtime_available: bool = True

    @property
    def default_machine(self) -> CliMachine:
        return self.machine_catalog.default


CLI_INSTALLATION = CliInstallationDefinition(
    machine_catalog=CLI_MACHINE_CATALOG,
    supply_graph_mode=GraphMode.CORE_SIMPLE,
    supply_tree_solver=solve_steiner_tree,
)
