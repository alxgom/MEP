import pytest

from mep_routing.installations.cli import (
    CliSizingContext, CoreConstantFrictionSizing, CoreSupplyGrilleSizing,
    size_tree_edges,
)


def test_core_sizing_matches_constant_friction_rounding():
    policy = CoreConstantFrictionSizing(CliSizingContext(1260.0, 0.06))

    full = policy.size(30.0, 30.0)
    partial = policy.size(12.0, 30.0)

    assert full.airflow_m3h == 1260.0
    assert full.height_mm == 150
    assert full.width_mm % 50 == 0
    assert partial.airflow_m3h == 504.0
    assert partial.width_mm < full.width_mm
    assert full.actual_velocity_m_s == pytest.approx(
        (1260.0 / 3600.0) / ((full.width_mm / 1000.0) * 0.15)
    )


def test_tree_sizing_accumulates_terminal_area_in_post_order():
    policy = CoreConstantFrictionSizing(CliSizingContext(1260.0, 0.06))
    sizes = size_tree_edges(
        ((0, 1), (1, 2), (1, 3)), 0, {2: 18.0, 3: 12.0}, policy,
    )

    assert sizes[(0, 1)].downstream_area_m2 == 30.0
    assert sizes[(0, 1)].airflow_m3h == 1260.0
    assert sizes[(1, 2)].airflow_m3h == 756.0
    assert sizes[(1, 3)].airflow_m3h == 504.0


def test_tree_sizing_policy_is_replaceable():
    class FixedPolicy:
        def size(self, downstream_area_m2, total_area_m2):
            return downstream_area_m2, total_area_m2

    sizes = size_tree_edges(((0, 1),), 0, {1: 8.0}, FixedPolicy())

    assert sizes[(0, 1)] == (8.0, 8.0)


def test_core_supply_grille_sizing_uses_room_airflow_and_face_velocity():
    policy = CoreSupplyGrilleSizing(1260.0)

    grille = policy.size(15.0, 30.0)

    assert grille.airflow_m3h == 630.0
    assert grille.width_mm == 800
    assert grille.height_mm == 150
    assert grille.face_velocity_m_s == pytest.approx(2.9166666667)
