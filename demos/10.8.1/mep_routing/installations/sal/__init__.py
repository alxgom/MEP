"""Sal ventilation catalog and route policy."""

from .catalog import SAL_OZEO_FLAT_MACHINE
from .controller import SalRoutingControllerContext, SalRoutingResult, solve_routing
from .routes import KITCHEN_ROUTE_NAME, LARGE_DUCT_ROUTE_NAMES, SHAFT_ROUTE_NAME
from .strategies import SalFlowContext, run_direct_small_pin_flow, run_sequential_routing, run_small_flow_stage, search_large_route_candidates, select_two_stage_routing

__all__ = [
    "KITCHEN_ROUTE_NAME",
    "LARGE_DUCT_ROUTE_NAMES",
    "SAL_OZEO_FLAT_MACHINE",
    "SalFlowContext",
    "SalRoutingControllerContext",
    "SalRoutingResult",
    "SHAFT_ROUTE_NAME",
    "run_sequential_routing",
    "run_direct_small_pin_flow",
    "run_small_flow_stage",
    "search_large_route_candidates",
    "select_two_stage_routing",
    "solve_routing",
]
