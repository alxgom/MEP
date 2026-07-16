from types import SimpleNamespace

from mep_routing.domain import MachineSpec
from mep_routing.installations.sal import SalLiveRoutingSession, SalSolverSettings


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
