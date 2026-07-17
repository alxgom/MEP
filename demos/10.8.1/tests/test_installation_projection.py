from types import SimpleNamespace

from mep_routing.ui.installation_projection import (
    InstallationProjection, ProjectedCapabilities, WorkbenchProjection, active_capabilities,
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


def test_disabled_active_installation_has_no_effective_capabilities():
    runtime = SimpleNamespace(capabilities=SimpleNamespace(machine_edit=True, solver_controls=True))
    app = SimpleNamespace(
        installation_header_state=SimpleNamespace(active="sal", enabled=frozenset()),
        routing_controller=SimpleNamespace(installations=SimpleNamespace(get=lambda key: runtime)),
    )
    capabilities = active_capabilities(app)
    assert not capabilities.machine_edit
    assert not capabilities.solver_controls
