from mep_routing.installations.sal.catalog import SAL_OZEO_FLAT_MACHINE
from mep_routing.installations.sal.definition import (
    SAL_GRAPH_MODES,
    SAL_HEURISTIC_MODE_LABELS,
    SAL_INSTALLATION,
    SAL_SEARCH_BACKEND_LABELS,
    SAL_STRATEGY_LABELS,
)
from mep_routing.installations.sal.orchestration import SalRoutingStrategy
from mep_routing.installations.sal.routes import KITCHEN_ROUTE_NAME, SHAFT_ROUTE_NAME
from mep_routing.routing.search import SearchBackend


def test_sal_definition_matches_current_machine_graph_and_solver_defaults():
    definition = SAL_INSTALLATION

    assert definition.machines == (SAL_OZEO_FLAT_MACHINE,)
    assert definition.default_machine is SAL_OZEO_FLAT_MACHINE
    assert definition.graph_modes == SAL_GRAPH_MODES
    assert definition.default_graph_mode == "Hannan Grid (numpy)"
    assert definition.routing_strategies == tuple(SalRoutingStrategy)
    assert definition.strategy_labels == SAL_STRATEGY_LABELS
    assert definition.default_routing_strategy is SalRoutingStrategy.FIRST_FIT
    assert definition.search_backends == tuple(SearchBackend)
    assert definition.search_backend_labels == SAL_SEARCH_BACKEND_LABELS
    assert definition.default_search_backend is SearchBackend.STATE_ASTAR
    assert definition.heuristic_mode_labels == SAL_HEURISTIC_MODE_LABELS
    assert definition.default_heuristic_mode == "Pin + bends"


def test_sal_definition_owns_current_route_classification_and_diameters():
    definition = SAL_INSTALLATION

    assert definition.is_large_route(SHAFT_ROUTE_NAME)
    assert definition.is_large_route(KITCHEN_ROUTE_NAME)
    assert not definition.is_large_route("Bathroom")
    assert definition.route_diameter_mm(SHAFT_ROUTE_NAME) == 125
    assert definition.route_diameter_mm("Bathroom") == 90
