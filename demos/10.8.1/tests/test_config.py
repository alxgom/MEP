from mep_routing.config import SALUBRIDAD_DEFAULTS
from mep_routing.config.legacy_keys import LEGACY_CONSTANT_TO_KEY


def test_sal_defaults_resolve_mapped_legacy_constants_to_semantic_keys():
    assert SALUBRIDAD_DEFAULTS.get_default("GRID_SPACING") == 200
    assert SALUBRIDAD_DEFAULTS.get_default("SALUBRIDAD.SOLVER.GRAPH.REGULAR_GRID_SPACING_MM") == 200
    assert LEGACY_CONSTANT_TO_KEY["GRID_SPACING"] == "SALUBRIDAD.SOLVER.GRAPH.REGULAR_GRID_SPACING_MM"


def test_sal_defaults_keep_solver_and_feasibility_values_distinct():
    assert SALUBRIDAD_DEFAULTS.get_default("ROUTING_WALL_CLEARANCE_MM") == 100
    assert SALUBRIDAD_DEFAULTS.get_default("C_BEND_DEFAULT") == 4000.0
