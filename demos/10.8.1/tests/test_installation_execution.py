from types import SimpleNamespace

from mep_routing.app import (
    InstallationCapabilities, InstallationCoordinator, InstallationRegistry,
    WorkbenchState,
)
from mep_routing.ui.installation_projection import InstallationVisual, installation_context_layers


def _result(name):
    return SimpleNamespace(
        routes=((name, [((0, 0), (100, 0))]),), status=name,
        elapsed_ms=1.0, total_nodes=2,
    )


def test_execution_respects_order_and_supplies_prior_context():
    contexts = {}
    class Runtime:
        capabilities = InstallationCapabilities()

        def __init__(self, key):
            self.key = key

        def solve(self, context):
            contexts[self.key] = context.context_obstacles
            return _result(self.key)

        def machine_pins(self, _state):
            return {"c_tl": (0, 0), "c_tr": (10, 0), "c_br": (10, 10), "c_bl": (0, 10)}

    state = WorkbenchState.create()
    coordinator = InstallationCoordinator(
        InstallationRegistry(Runtime(key) for key in ("sal", "cli")), state,
    )
    execution = coordinator.execute(("sal", "cli"), {"sal", "cli"}, "cli", object())
    assert not contexts["sal"]
    assert len(contexts["cli"]) == 2
    assert list(execution.runs) == ["sal", "cli"]
    assert state.installation("cli").result is execution.active_result
    assert execution.routing_tuple()[1] == "cli"


def test_coordinator_skips_unregistered_and_clears_disabled_results():
    class Runtime:
        key = "sal"
        capabilities = InstallationCapabilities(terminal_tools=True)
        def solve(self, _context):
            return _result("sal")
        def machine_pins(self, _state):
            return {}

    state = WorkbenchState.create()
    state.installation("cli").result = _result("stale")
    coordinator = InstallationCoordinator(InstallationRegistry((Runtime(),)), state)
    execution = coordinator.execute(("cli", "sal"), {"cli", "sal"}, "cli", object())
    assert list(execution.runs) == ["sal"]
    assert state.installation("cli").result is None
    assert execution.active_result is None
    assert execution.active_status == "Routing unavailable: Cli"


def test_context_projection_excludes_active_installation():
    runs = {"sal": _result("sal"), "cli": _result("cli")}
    visuals = {key: InstallationVisual({}, {}) for key in runs}
    layers = installation_context_layers(
        ("sal", "cli"), {"sal", "cli"}, "cli", runs, visuals,
    )
    assert [layer.installation for layer in layers] == ["sal"]
