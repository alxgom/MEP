from types import SimpleNamespace

from mep_routing.app.installation_runtime import CliInstallationRuntime
from mep_routing.ui.demo_facade import DemoUiFacade
from mep_routing.ui.installation_projection import (
    InstallationProjection, ProjectedCapabilities, WorkbenchProjection, active_capabilities,
    project_workbench,
)


def _item(key, active):
    result = SimpleNamespace(routes=((f"{key}-route", ()),))
    return InstallationProjection(
        key, active, ProjectedCapabilities(machine_edit=True, terminal_tools=key == "sal"),
        result, SimpleNamespace(), {f"{key}-terminal": (1, 2)}, (0, 0, 0), {},
    )


def test_projection_keeps_all_enabled_results_visible_but_only_active_interactive():
    sal, cli = _item("sal", True), _item("cli", False)
    projection = WorkbenchProjection(sal, (sal, cli))

    assert [item.key for item in projection.visible] == ["sal", "cli"]
    assert sal.can("terminal_tools")
    assert not cli.can("machine_edit")
    assert [layer.installation for layer in projection.context_layers] == ["cli"]


def test_cli_projection_builds_physical_width_grille_segment_from_wall_normal():
    connection = SimpleNamespace(
        route_name="Living Supply", grille_point=(1000, 2000),
        inward_normal=(0.0, 1.0), width_mm=800,
    )
    item = InstallationProjection(
        "cli", True, ProjectedCapabilities(),
        SimpleNamespace(routes=(), grille_plan=SimpleNamespace(connections=(connection,))),
        SimpleNamespace(), {"Living Supply": connection.grille_point}, (0, 0, 0), {},
    )

    assert item.terminal_segments["Living Supply"] == (
        (600.0, 2000.0), (1400.0, 2000.0),
    )


def test_disabled_active_installation_has_no_effective_capabilities():
    runtime = SimpleNamespace(capabilities=SimpleNamespace(machine_edit=True, solver_controls=True))
    app = SimpleNamespace(
        installation_header_state=SimpleNamespace(active="sal", enabled=frozenset()),
        routing_controller=SimpleNamespace(installations=SimpleNamespace(get=lambda key: runtime)),
    )
    capabilities = active_capabilities(app)
    assert not capabilities.machine_edit
    assert not capabilities.solver_controls


def test_disabled_active_installation_projects_no_machine():
    runtime = SimpleNamespace(
        capabilities=SimpleNamespace(),
        machine_pins=lambda _state: {"c_tl": (0, 0)},
    )
    state = SimpleNamespace(pose=SimpleNamespace(as_tuple=lambda: (10, 20, 0)), result=None)
    app = SimpleNamespace(
        installation_header_state=SimpleNamespace(
            active="cli", enabled=frozenset(), order=("cli",),
        ),
        routing_controller=SimpleNamespace(
            installations=SimpleNamespace(get=lambda _key: runtime),
        ),
        workbench_state=SimpleNamespace(
            dwelling=SimpleNamespace(workspace=object()),
            installations={"cli": state},
        ),
        terminals={},
    )

    projection = project_workbench(app)

    assert projection.visible == ()
    assert projection.active.pins == {}


def test_cli_advertises_its_own_routing_weight_projection():
    assert CliInstallationRuntime.capabilities.routing_weights


def test_weight_overlay_draw_uses_explicit_active_workspace():
    calls = []
    canvas = SimpleNamespace(
        draw_edge_heatmap=lambda *args: calls.append(args),
    )
    stale_sal = SimpleNamespace(
        overlay=SimpleNamespace(values={(0, 1): 9.0}), env="sal-env",
    )
    active_cli = SimpleNamespace(
        overlay=SimpleNamespace(values={}), env="cli-env",
    )
    app = SimpleNamespace(
        edge_weight_heatmap_enabled=True, canvas_ui=canvas,
        workbench_state=SimpleNamespace(dwelling=SimpleNamespace(workspace=stale_sal)),
        demo_config=SimpleNamespace(overlap_block_weight=1e6),
        theme=SimpleNamespace(blocked_edge=(1, 2, 3)),
    )

    DemoUiFacade(app, None).draw_edge_weight_heatmap("screen", active_cli)

    assert calls[0][1] == {}
    assert calls[0][2] == "cli-env"
