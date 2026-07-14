"""Concrete capability definition for the current Sal installation.

This is intentionally a compact description of Demo 10.8.1 behavior.  It is
not a generic installation framework; consumers can use the definition as the
single seam for selecting Sal machines, graphs, and solver strategies.
"""

from dataclasses import dataclass

from mep_routing.domain import MachineSpec
from mep_routing.routing.search import SearchBackend

from .catalog import SAL_OZEO_FLAT_MACHINE
from .orchestration import SalRoutingStrategy
from .routes import LARGE_DUCT_ROUTE_NAMES


SAL_GRAPH_MODES = (
    "Regular 200mm Grid",
    "Hannan Grid (numpy)",
    "Epsilon Grid (core-like numpy)",
)

SAL_STRATEGY_LABELS = (
    "Greedy (Dual-Sort)",
    "First Fit",
    "Best Fit",
    "Negotiated Congestion",
    "Negotiated Congestion (Favour Large)",
    "Min-Cost Flow (Small Pins)",
    "Min-Cost Flow (Two-Stage)",
)

SAL_SEARCH_BACKEND_LABELS = (
    "State-expanded A*",
    "Line graph L(G) A*",
    "Line graph L(G) GBFS",
)

SAL_HEURISTIC_MODE_LABELS = (
    "Pin + bends",
    "Pin distance",
    "Machine envelope",
    "Zero",
)


@dataclass(frozen=True)
class SalInstallationDefinition:
    """Machines and routing capabilities supported by Sal in this demo."""

    machines: tuple[MachineSpec, ...]
    graph_modes: tuple[str, ...]
    routing_strategies: tuple[SalRoutingStrategy, ...]
    strategy_labels: tuple[str, ...]
    search_backends: tuple[SearchBackend, ...]
    search_backend_labels: tuple[str, ...]
    heuristic_mode_labels: tuple[str, ...]
    large_route_names: frozenset[str]
    default_machine: MachineSpec
    default_graph_mode: str
    default_routing_strategy: SalRoutingStrategy
    default_search_backend: SearchBackend
    default_heuristic_mode: str

    def is_large_route(self, route_name: str) -> bool:
        return route_name in self.large_route_names

    def route_diameter_mm(self, route_name: str) -> int:
        return self.default_machine.route_diameter_mm(route_name)


SAL_INSTALLATION = SalInstallationDefinition(
    machines=(SAL_OZEO_FLAT_MACHINE,),
    graph_modes=SAL_GRAPH_MODES,
    routing_strategies=tuple(SalRoutingStrategy),
    strategy_labels=SAL_STRATEGY_LABELS,
    search_backends=tuple(SearchBackend),
    search_backend_labels=SAL_SEARCH_BACKEND_LABELS,
    heuristic_mode_labels=SAL_HEURISTIC_MODE_LABELS,
    large_route_names=LARGE_DUCT_ROUTE_NAMES,
    default_machine=SAL_OZEO_FLAT_MACHINE,
    default_graph_mode=SAL_GRAPH_MODES[1],
    default_routing_strategy=SalRoutingStrategy.FIRST_FIT,
    default_search_backend=SearchBackend.STATE_ASTAR,
    default_heuristic_mode=SAL_HEURISTIC_MODE_LABELS[0],
)
