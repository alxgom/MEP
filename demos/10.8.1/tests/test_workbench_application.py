from types import SimpleNamespace

import pytest

from mep_routing.app import DemoWorkbenchApplication, WorkbenchState


def test_application_owns_controllers_without_flattening_callbacks():
    source = SimpleNamespace(
        workbench_state=WorkbenchState.create(),
        installation_header_state=SimpleNamespace(active="sal"),
    )
    app = DemoWorkbenchApplication(source).bind(
        routing=SimpleNamespace(), dwelling=SimpleNamespace(), ui=SimpleNamespace(),
        commands=SimpleNamespace(), observation=SimpleNamespace(runtime="runtime", history="history", logs="logs"),
        chrome=SimpleNamespace(),
    )

    assert app.observation_runtime == "runtime"
    assert app.get_active_machine_pose() == (0.0, 0.0, 0.0)
    with pytest.raises(AttributeError):
        _ = app.build_grid


def test_application_mutates_remaining_composition_data_in_one_place():
    source = SimpleNamespace(show_heatmap=False)
    app = DemoWorkbenchApplication(source)

    app.show_heatmap = True

    assert source.show_heatmap is True
