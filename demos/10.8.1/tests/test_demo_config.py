from dataclasses import FrozenInstanceError

import pytest

from mep_routing.app import build_demo_config
from mep_routing.installations.sal import SAL_INSTALLATION
from mep_routing.ui import WorkbenchTheme


def test_demo_config_builds_runtime_value_objects_from_typed_defaults():
    config = build_demo_config(SAL_INSTALLATION)

    assert config.graph_types == tuple(SAL_INSTALLATION.graph_modes)
    assert config.common_graph.grid_spacing_mm == config.grid_spacing_mm
    assert config.terminal_runtime.remap_tolerance_mm == config.preferred_terminal_remap_tolerance_mm
    with pytest.raises(FrozenInstanceError):
        config.fps = 30


def test_workbench_theme_keeps_layout_and_palette_out_of_runtime_state():
    theme = WorkbenchTheme()

    assert theme.panel_width == 280
    assert theme.route_colors["Supply Air Tree"] == (52, 152, 219)
    with pytest.raises(TypeError):
        theme.route_colors["Supply Air Tree"] = (0, 0, 0)
