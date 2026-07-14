from mep_routing.installations.sal.catalog import SAL_OZEO_FLAT_MACHINE
from mep_routing.installations.sal.definition import (
    SAL_GRAPH_MODES,
    SAL_INSTALLATION,
    SAL_STRATEGY_LABELS,
)
from mep_routing.installations.sal.orchestration import SalRoutingStrategy
from mep_routing.installations.sal.routes import KITCHEN_ROUTE_NAME, SHAFT_ROUTE_NAME


def test_sal_definition_matches_current_machine_graph_and_solver_defaults():
    definition = SAL_INSTALLATION

    assert definition.machines == (SAL_OZEO_FLAT_MACHINE,)
    assert definition.default_machine is SAL_OZEO_FLAT_MACHINE
    assert definition.graph_modes == SAL_GRAPH_MODES
    assert definition.default_graph_mode == "Hannan Grid (numpy)"
    assert definition.routing_strategies == tuple(SalRoutingStrategy)
    assert definition.strategy_labels == SAL_STRATEGY_LABELS
    assert definition.default_routing_strategy is SalRoutingStrategy.FIRST_FIT


def test_sal_definition_owns_current_route_classification_and_diameters():
    definition = SAL_INSTALLATION

    assert definition.is_large_route(SHAFT_ROUTE_NAME)
    assert definition.is_large_route(KITCHEN_ROUTE_NAME)
    assert not definition.is_large_route("Bathroom")
    assert definition.route_diameter_mm(SHAFT_ROUTE_NAME) == 125
    assert definition.route_diameter_mm("Bathroom") == 90
