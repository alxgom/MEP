from types import SimpleNamespace

from mep_routing.domain import MachineSpec
from mep_routing.installations.sal import (
    SalLiveRoutingCallbacks,
    SalLiveRoutingSession,
    SalSolverSettings,
)


def test_live_routing_session_uses_one_policy_machine_and_workspace_snapshot():
    machine = MachineSpec(
        name="Alternate",
        body_width_mm=300,
        body_height_mm=200,
        overall_width_mm=350,
        small_duct_diameter_mm=80,
        large_duct_diameter_mm=140,
        small_pin_stub_length_mm=50,
        large_pin_stub_length_mm=100,
        large_route_names=frozenset({"Shaft"}),
    )
    cache, overlay = object(), object()
    workspace = SimpleNamespace(
        edge_list=((0, 1),), edge_coords=object(),
        static_clearance_cache=cache, overlay=overlay,
    )
    settings = SalSolverSettings(1200, 2.5, 0.25, 100, 50, 1e9, 1.1, 2, 3)
    installation = SimpleNamespace(search_backends=("first", "selected"))
    env = SimpleNamespace(nodes=((0, 0), (1, 0)))
    session = SalLiveRoutingSession(
        installation, machine, workspace, object(), (), (), (), (),
        (30, 40), 90, settings, 1, lambda *_args: 0,
    )

    runtime = session.routing_runtime(env)
    assert session.policy.crossing_penalty == 3000
    assert runtime.env is env
    assert runtime.cache is cache and runtime.overlay is overlay
    assert runtime.search_backend == "selected"
    assert runtime.heuristic_mode == 2 and runtime.bend_cost == 1200
    assert runtime.context.machine_overall_width_mm == 350
    assert runtime.context.machine_body_height_mm == 200
    assert runtime.context.route_diameter("Shaft") == 140
    assert session.route_analysis().route_diameter("Shaft") == 140


def test_live_session_derives_adapter_machine_and_workspace_hooks():
    workspace = SimpleNamespace(grid_available=True, env=object())
    machine = SimpleNamespace(
        center=(10, 20), angle=90, pins={"pin": (1, 2)},
        preflight_error=lambda: None, refresh_graph=lambda _pins: None,
        snap_pins=lambda _pins: {},
    )
    spec = SimpleNamespace(
        small_duct_diameter_mm=80, large_duct_diameter_mm=140,
        route_diameter_mm=lambda _name: 80,
    )
    settings = SalSolverSettings(1, 1, 0, 0, 0, 99, 1, 0, 0)
    session = SimpleNamespace(
        installation=object(), machine_spec=spec, workspace=workspace,
        machine_center=(10, 20), machine_angle=90, settings=settings,
        routing_runtime=lambda _env, _policy: object(),
    )
    callback = lambda *_args, **_kwargs: None
    callbacks = SalLiveRoutingCallbacks(*([callback] * 12))

    adapter = SalLiveRoutingSession.application_adapter(session, machine, {}, callbacks)
    assert adapter.machine_center == (10, 20) and adapter.machine_angle == 90
    assert adapter.small_diameter == 80 and adapter.large_diameter == 140
    assert adapter.hooks.grid_available() is True
    assert adapter.hooks.machine_pins() == {"pin": (1, 2)}
