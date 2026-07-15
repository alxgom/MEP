from types import SimpleNamespace

from shapely.geometry import box

from mep_routing.domain import MachineSpec
from mep_routing.routing import MachineRoutingSession


def _spec():
    return MachineSpec(
        name="Test",
        body_width_mm=100,
        body_height_mm=60,
        overall_width_mm=140,
        small_duct_diameter_mm=20,
        large_duct_diameter_mm=40,
        small_pin_stub_length_mm=15,
        large_pin_stub_length_mm=25,
        large_route_names=frozenset({"Large"}),
    )


def _session(**overrides):
    values = dict(
        spec=_spec(),
        center=(100.0, 100.0),
        angle=0,
        routing_region=box(0, 0, 500, 500),
        columns=(),
        workspace=SimpleNamespace(spatial_index=None),
        graph_type=0,
        update_dynamic_env=lambda _polygon: None,
        build_grid=lambda **_kwargs: None,
    )
    values.update(overrides)
    return MachineRoutingSession(**values)


def test_machine_session_preflight_uses_selected_machine_geometry():
    assert _session().preflight_error() is None
    assert _session(center=(600.0, 100.0)).preflight_error() == "Blocked: Machine outside region"
    assert _session(columns=(box(90, 90, 110, 110),)).preflight_error() == "Blocked: Machine collides with column"


def test_machine_session_refresh_rebuilds_only_non_regular_graphs():
    calls = []
    _session(
        graph_type=2,
        build_grid=lambda **kwargs: calls.append(("grid", kwargs)),
        update_dynamic_env=lambda polygon: calls.append(("dynamic", polygon)),
    ).refresh_graph(_session().pins)
    assert [name for name, _value in calls] == ["grid", "dynamic"]


def test_machine_session_snaps_port_access_to_current_spatial_index():
    spatial_index = SimpleNamespace(query=lambda _point: (0.0, 7))
    session = _session(workspace=SimpleNamespace(spatial_index=spatial_index))
    snapped = session.snap_pins(session.pins)
    assert snapped
    assert {target["node_idx"] for targets in snapped.values() for target in targets} == {7}
    assert session.spec.route_diameter_mm("Large") == 40
