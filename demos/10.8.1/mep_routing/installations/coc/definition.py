"""Coc capabilities for dwelling-level kitchen exhaust routing."""

from __future__ import annotations

from dataclasses import dataclass

from mep_routing.domain import Catalog, CatalogEntry
from mep_routing.graphs import GraphMode


@dataclass(frozen=True)
class CocSource:
    key: str = "exported_exhaust"


@dataclass(frozen=True)
class CocInstallationDefinition:
    source_catalog: Catalog[CocSource]
    graph_modes: tuple[GraphMode, ...] = (GraphMode.CORE_SIMPLE, GraphMode.REGULAR)
    graph_labels: tuple[str, ...] = (
        "Routing-Core Port: Simple Grid", "Regular 200mm Grid",
    )
    strategy_labels: tuple[str, ...] = ("Two-terminal exhaust",)
    search_backend_labels: tuple[str, ...] = ("Kou shortest connection",)
    heuristic_mode_labels: tuple[str, ...] = ("Shortest-path distance",)
    start_mode_labels: tuple[str, ...] = ("Exported connector to Coc shaft",)


COC_INSTALLATION = CocInstallationDefinition(Catalog(
    (CatalogEntry(
        "exported_exhaust", "Exported kitchen exhaust", CocSource(),
        "Fixed appliance connector from dwelling export",
    ),),
    "exported_exhaust",
))
