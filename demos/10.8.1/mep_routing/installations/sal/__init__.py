"""Sal ventilation catalog and route policy."""

from .catalog import SAL_OZEO_FLAT_MACHINE
from .controller import SalRoutingControllerContext, SalRoutingResult, solve_routing
from .application import SalApplicationAdapter, SalApplicationHooks, SalSolverSettings
from .live_solver import SalLiveRoutingSession
from .definition import SAL_INSTALLATION, SalInstallationDefinition
from .routes import KITCHEN_ROUTE_NAME, LARGE_DUCT_ROUTE_NAMES, SHAFT_ROUTE_NAME
from .route_plan import SalRoutePlan, build_sal_route_plan
from .policy import SalSolverPolicy
from .prepared import SalPreparedRoutingProblem
from .strategy_dispatch import SalStrategyOutcome, SalStrategyRuntime, solve_prepared_strategy
from .strategies import SalFlowContext, run_direct_small_pin_flow, run_sequential_routing, run_small_flow_stage, search_large_route_candidates, select_two_stage_routing

__all__ = [
    "KITCHEN_ROUTE_NAME",
    "LARGE_DUCT_ROUTE_NAMES",
    "SAL_OZEO_FLAT_MACHINE",
    "SAL_INSTALLATION",
    "SalInstallationDefinition",
    "SalFlowContext",
    "SalRoutingControllerContext",
    "SalApplicationAdapter",
    "SalApplicationHooks",
    "SalSolverSettings",
    "SalLiveRoutingSession",
    "SalRoutingResult",
    "SalRoutePlan",
    "SalSolverPolicy",
    "SalPreparedRoutingProblem",
    "SalStrategyOutcome",
    "SalStrategyRuntime",
    "SHAFT_ROUTE_NAME",
    "build_sal_route_plan",
    "run_sequential_routing",
    "run_direct_small_pin_flow",
    "run_small_flow_stage",
    "search_large_route_candidates",
    "select_two_stage_routing",
    "solve_routing",
    "solve_prepared_strategy",
]
