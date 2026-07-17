from types import SimpleNamespace

from mep_routing.app import WorkbenchSmokeResult, run_workbench_smoke


class _DwellingRuntime:
    def __init__(self, application):
        self.application = application

    def generate(self):
        self.application.workbench_state.dwelling.active = object()


def test_workbench_smoke_uses_application_controller_boundaries():
    application = SimpleNamespace(
        current_scenario_label="smoke dwelling",
        workbench_state=SimpleNamespace(
            dwelling=SimpleNamespace(active=None),
        ),
        routing_controller=SimpleNamespace(
            solve_installations=lambda: ({"sal": [(0, 0), (1, 1)]}, "ok", 4.5, 12),
        ),
    )
    application.dwelling_runtime = _DwellingRuntime(application)

    result = run_workbench_smoke(application)

    assert result == WorkbenchSmokeResult(
        dwelling_label="smoke dwelling",
        route_count=1,
        status="ok",
        elapsed_ms=4.5,
        total_nodes=12,
    )
