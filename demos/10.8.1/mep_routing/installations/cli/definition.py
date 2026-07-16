"""Declared Cli capabilities available before its interactive runtime is enabled."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mep_routing.domain import Catalog
from mep_routing.graphs import GraphMode
from mep_routing.routing import SteinerTreeResult, solve_steiner_tree

from .catalog import CLI_MACHINE_CATALOG, CliMachine


@dataclass(frozen=True)
class CliInstallationDefinition:
    """Cli contracts that are implemented without claiming end-to-end routing."""

    machine_catalog: Catalog[CliMachine]
    supply_graph_mode: GraphMode
    supply_tree_solver: Callable[..., SteinerTreeResult | None]
    interactive_runtime_available: bool = False

    @property
    def default_machine(self) -> CliMachine:
        return self.machine_catalog.default


CLI_INSTALLATION = CliInstallationDefinition(
    machine_catalog=CLI_MACHINE_CATALOG,
    supply_graph_mode=GraphMode.CORE_SIMPLE,
    supply_tree_solver=solve_steiner_tree,
)
