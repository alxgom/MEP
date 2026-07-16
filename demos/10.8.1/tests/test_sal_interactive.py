from types import SimpleNamespace

from mep_routing.installations.sal import SalInteractiveCallbacks, SalInteractiveSolver


class Overlay:
    def __init__(self):
        self.excluded_edges = set()
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        self.excluded_edges.clear()


def test_interactive_solver_owns_machine_adapter_and_overlay_lifecycle():
    overlay = Overlay()
    workspace = SimpleNamespace(overlay=overlay, spatial_index=None)
    result = SimpleNamespace(excluded_overlay_edges={(1, 2)})
    captured = {}
    analysis = SimpleNamespace(
        score=lambda routes, crossings: 42,
        conflict_summary=lambda routes: "none",
    )

    def application_adapter(machine_session, terminals, callbacks):
        captured.update(machine=machine_session, terminals=terminals, callbacks=callbacks)
        return SimpleNamespace(solve=lambda: result)

    live_session = SimpleNamespace(
        machine_spec=object(),
        machine_center=(10, 20),
        machine_angle=90,
        routing_region=object(),
        workspace=workspace,
        route_analysis=lambda shaft: analysis,
        application_adapter=application_adapter,
    )
    callback = lambda *_args, **_kwargs: None
    callbacks = SalInteractiveCallbacks(*([callback] * 9))
    materializer = SimpleNamespace(
        source_start_nodes=callback,
        route_segments=callback,
        build_routes=callback,
    )

    actual = SalInteractiveSolver(
        live_session,
        {"Kitchen": (1, 2)},
        2,
        (object(),),
        (object(),),
        materializer,
        object(),
        callbacks,
    ).solve()

    assert actual is result
    assert overlay.reset_calls == 1
    assert overlay.excluded_edges == {(1, 2)}
    assert captured["terminals"] == {"Kitchen": (1, 2)}
    assert captured["machine"].center == (10, 20)
    assert captured["machine"].graph_type == 2
    assert captured["callbacks"].source_start_nodes is materializer.source_start_nodes
    assert captured["callbacks"].score_routes([], 0, object()) == 42
