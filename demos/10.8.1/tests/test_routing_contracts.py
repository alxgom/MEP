from dataclasses import FrozenInstanceError

import pytest

from mep_routing.routing.contracts import (
    RoutingProblem,
    RoutingRequest,
    SolvedRoute,
    SolverFailure,
    SolverResult,
)


def test_problem_preserves_solver_inputs_without_installation_dependencies():
    target = {"node_idx": 8, "in_dir": "E", "pin": "tl"}
    request = RoutingRequest("wet-room", (1, 2), (target,))
    graph = object()
    problem = RoutingProblem(graph, (request,), {(1, 2): 12.5})

    assert problem.graph is graph
    assert problem.requests == (request,)
    assert problem.requests[0].target_candidates[0] is target
    assert problem.edge_weights[(1, 2)] == 12.5
    with pytest.raises(FrozenInstanceError):
        request.key = "other"


def test_problem_rejects_ambiguous_or_incomplete_requests():
    with pytest.raises(ValueError, match="source node"):
        RoutingRequest("bathroom", (), (2,))
    with pytest.raises(ValueError, match="target candidate"):
        RoutingRequest("bathroom", (1,), ())
    with pytest.raises(ValueError, match="unique"):
        request = RoutingRequest("bathroom", (1,), (2,))
        RoutingProblem(object(), (request, request))


def test_result_has_structured_success_and_failure_outcomes():
    success = SolverResult(
        routes=(SolvedRoute("bathroom", (1, 3, 8)),),
        elapsed_ms=4.5,
        route_node_count=12,
        objective_cost=18.0,
        completed_request_count=1,
    )
    failure = SolverResult(
        failure=SolverFailure("no_path", "No route reaches a target", "bathroom")
    )

    assert success.success
    assert success.routes[0].route == (1, 3, 8)
    assert success.route_node_count == 12
    assert success.objective_cost == 18.0
    assert success.completed_request_count == 1
    assert not failure.success
    assert failure.failure.request_key == "bathroom"
    with pytest.raises(ValueError, match="both routes and a failure"):
        SolverResult(success.routes, failure.failure)
