from dataclasses import FrozenInstanceError

import pytest

from mep_routing.installations.sal.policy import SalSolverPolicy
from mep_routing.installations.sal.prepared import SalPreparedRoutingProblem
from mep_routing.installations.sal.route_plan import build_sal_route_plan


def test_prepared_sal_problem_normalizes_only_shaft_boundary_nodes():
    route_plan = build_sal_route_plan({"Bathroom": (10, 20)}, (0, 0))
    policy = SalSolverPolicy(4_000, 5, 1.05, 200, 150, 1e9, 1.05, 0)
    global_pins = {"left_mid": (1, 2)}
    pin_node_map = {"left_mid": [{"node_idx": 3}]}
    shaft_path = [7, 8, 9]
    chosen_target = {"pin": "left_mid", "node_idx": 3}

    prepared = SalPreparedRoutingProblem(
        route_plan=route_plan,
        policy=policy,
        global_pins=global_pins,
        pin_node_map=pin_node_map,
        shaft_boundary_nodes=[4, 5],
        shaft_node_idx=6,
        shaft_path=shaft_path,
        chosen_shaft_pin="left_mid",
        chosen_shaft_target=chosen_target,
    )

    assert prepared.shaft_boundary_nodes == (4, 5)
    assert prepared.route_plan is route_plan
    assert prepared.policy is policy
    assert prepared.global_pins is global_pins
    assert prepared.pin_node_map is pin_node_map
    assert prepared.shaft_path is shaft_path
    assert prepared.chosen_shaft_target is chosen_target


def test_prepared_sal_problem_is_frozen():
    prepared = SalPreparedRoutingProblem(
        route_plan=build_sal_route_plan({}, (0, 0)),
        policy=SalSolverPolicy(4_000, 5, 1.05, 200, 150, 1e9, 1.05, 0),
        global_pins={},
        pin_node_map={},
        shaft_boundary_nodes=(),
        shaft_node_idx=0,
        shaft_path=[],
        chosen_shaft_pin="left_mid",
        chosen_shaft_target={},
    )

    with pytest.raises(FrozenInstanceError):
        prepared.shaft_node_idx = 1
