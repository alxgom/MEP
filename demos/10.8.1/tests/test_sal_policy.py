from dataclasses import FrozenInstanceError

import pytest

from mep_routing.installations.sal.policy import SalSolverPolicy


def _current_policy(**overrides):
    values = {
        "bend_cost": 4_000.0,
        "crossing_penalty_multiplier": 5.0,
        "duct_buffer_ratio": 1.05,
        "shaft_clearance_mm": 200.0,
        "machine_clearance_soft_margin_mm": 150.0,
        "overlap_block_weight": 1e9,
        "min_piece_factor": 1.05,
        "heuristic_mode": 0,
    }
    values.update(overrides)
    return SalSolverPolicy(**values)


def test_sal_solver_policy_preserves_current_derived_penalty_relationships():
    policy = _current_policy()

    assert policy.crossing_penalty == 20_000.0
    assert policy.clearance_penalty == 20_000.0
    assert policy.overlap_score_penalty == 200_000.0
    assert policy.short_piece_score_penalty == 8_000.0


def test_sal_solver_policy_preserves_current_negotiated_defaults():
    policy = _current_policy()

    assert policy.negotiated_iterations == 20
    assert policy.negotiated_present_penalty == 20_000.0
    assert policy.negotiated_history_penalty == 4_000.0
    assert policy.negotiated_large_route_factor == 0.35


def test_sal_solver_policy_is_an_immutable_runtime_snapshot():
    policy = _current_policy(heuristic_mode=2, min_piece_factor=1.25)

    assert policy.heuristic_mode == 2
    assert policy.min_piece_factor == 1.25
    with pytest.raises(FrozenInstanceError):
        policy.bend_cost = 5_000.0
