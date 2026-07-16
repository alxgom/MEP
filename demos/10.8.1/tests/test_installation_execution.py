from types import SimpleNamespace

from mep_routing.app import InstallationRunSpec, execute_installations
from mep_routing.ui.installation_projection import InstallationVisual, installation_context_layers


def _result(name):
    return SimpleNamespace(
        routes=((name, [((0, 0), (100, 0))]),), status=name,
        elapsed_ms=1.0, total_nodes=2,
    )


def test_execution_respects_order_and_supplies_prior_context():
    contexts = {}
    def solve(key, context):
        contexts[key] = context
        return _result(key)

    specs = {
        key: InstallationRunSpec(
            lambda context, key=key: solve(key, context),
            lambda: {"c_tl": (0, 0), "c_tr": (10, 0), "c_br": (10, 10), "c_bl": (0, 10)},
        ) for key in ("sal", "cli")
    }
    execution = execute_installations(("sal", "cli"), {"sal", "cli"}, "cli", specs)
    assert not contexts["sal"]
    assert len(contexts["cli"]) == 2
    assert list(execution.runs) == ["sal", "cli"]
    assert execution.routing_tuple()[1] == "cli"


def test_context_projection_excludes_active_installation():
    runs = {"sal": _result("sal"), "cli": _result("cli")}
    visuals = {key: InstallationVisual({}, {}) for key in runs}
    layers = installation_context_layers(
        ("sal", "cli"), {"sal", "cli"}, "cli", runs, visuals,
    )
    assert [layer.installation for layer in layers] == ["sal"]
