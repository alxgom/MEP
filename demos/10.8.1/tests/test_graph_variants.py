from shapely.geometry import box

from mep_routing.graphs.variants import build_epsilon_variant, build_hannan_variant


def test_hannan_variant_preserves_terminal_and_machine_access_axes():
    region = box(0, 0, 1000, 1000)
    template = {
        "xs": {100, 500, 900}, "ys": {100, 500, 900},
        "preserve_x": set(), "preserve_y": set(),
        "priority_x": set(), "priority_y": set(),
    }

    result = build_hannan_variant(
        template=template, allowed_region=region, node_region=region, wall_polys=[], wall_thickness_mm=20,
        terminals={"Wet": (500, 500)}, shaft_extraction=None, machine_access_points=[(300, 500)],
    )

    assert (500, 500) in result.required_points
    assert 300 in result.axes_x
    assert (300.0, 500.0) in {tuple(node) for node in result.nodes}
    assert result.edges


def test_epsilon_variant_keeps_terminal_as_a_required_graph_node():
    region = box(0, 0, 1000, 1000)

    result = build_epsilon_variant(
        allowed_region=region, node_region=region, covers=[], columns=[], shafts=[], wall_polys=[], wall_thickness_mm=20,
        terminals={"Wet": (500, 500)}, shaft_core_entry_specs=[], shaft_extraction=None,
        machine_access_points=[], epsilon_mm=100, scaffold_spacing_mm=400,
    )

    assert (500, 500) in result.required_points
    assert (500.0, 500.0) in {tuple(node) for node in result.nodes}
    assert result.edges
