from types import SimpleNamespace

from shapely.geometry import Polygon

from mep_routing.ui.frame import (
    CanvasFrameCallbacks,
    CanvasFramePalette,
    CanvasFrameState,
    SidebarFrameState,
    active_selected_route,
    build_canvas_scene,
    build_sidebar_view,
)


def _palette():
    colors = [(index, index, index) for index in range(16)]
    return CanvasFramePalette(*colors)


def _pins():
    return {
        "c_tl": (0, 0), "c_tr": (2, 0), "c_br": (2, 2), "c_bl": (0, 2),
        "tl": (0, 0), "tr": (2, 0), "bl": (0, 2), "br": (2, 2),
        "left_mid": (0, 1), "right_mid": (2, 1),
    }


def test_active_selected_route_discards_stale_selection():
    routes = (("Kitchen", ()),)
    assert active_selected_route("Kitchen", routes) == "Kitchen"
    assert active_selected_route("Bathroom", routes) is None
    assert active_selected_route("Kitchen", ()) is None


def test_canvas_builder_derives_selection_graph_machine_and_guides():
    room = SimpleNamespace(name="Kitchen", has_cover=True, polygon=Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]))
    other = SimpleNamespace(name="Hall", has_cover=False, polygon=Polygon([(5, 0), (8, 0), (8, 4), (5, 4)]))
    shaft = Polygon([(8, 0), (9, 0), (9, 1), (8, 1)])
    env = SimpleNamespace(nodes=((0, 0), (1, 0)), adj={0: ((1, 1, 0),), 1: ((0, 1, 0),)})
    node_indices = {point: index for index, point in enumerate(_pins().values())}
    callbacks = CanvasFrameCallbacks(
        to_screen=lambda x, y: (int(x * 10), int(y * 10)),
        representative_point=lambda geometry: (8.5, 0.5),
        route_draw_width=lambda name: 5 if name == "Shaft" else 3,
        spatial_query=lambda point: (0.0, node_indices[point]),
    )
    fields = {
        "Shaft": {node_indices[(0, 1)]: 1, node_indices[(2, 1)]: 5},
        "Bathroom": {node_indices[(0, 0)]: 2, node_indices[(2, 0)]: 4},
    }
    state = CanvasFrameState(
        rooms=(room, other), doors=({"d1": (1, 0), "d2": (2, 0)},),
        columns=(Polygon([(6, 1), (7, 1), (7, 2), (6, 2)]),),
        shafts=(shaft,), shaft_extraction=shaft, graph_env=env, show_grid_graph=True,
        terminals={"Kitchen": (3, 3), "Bathroom": (6, 3)},
        routes=(("Kitchen", (((0, 0), (3, 3)),)),), selected_route_name="Kitchen",
        selected_room_polygon=room.polygon, global_pins=_pins(), selected_pins={"right_mid"},
        auto_placement_mode=1, placement_fields=fields, wet_room_names=("Kitchen", "Bathroom"),
        route_colors={"Kitchen": (20, 30, 40), "Bathroom": (50, 60, 70)},
        palette=_palette(), callbacks=callbacks, room_wall_width=4,
        terminal_area_start=(1, 1), terminal_area_end=(2, 2),
    )

    scene = build_canvas_scene(state)

    assert scene.rooms[0].color == state.palette.covered_room
    assert scene.rooms[1].color == state.palette.deselected_room
    assert scene.grid_edges == [((0, 0), (10, 0))]
    assert scene.routes[0].selected and scene.routes[0].width == 3
    assert scene.machine.outline == [(0, 0), (20, 0), (20, 20), (0, 20)]
    assert scene.guide_lines[0].start == (0, 10)  # lower-cost left exhaust pin
    assert scene.guide_lines[1].start == (20, 10)  # opposite kitchen pin
    assert scene.guide_lines[2].color == (50, 60, 70)


def test_sidebar_builder_formats_display_modes_and_frame_name():
    state = SidebarFrameState(
        auto_placement_mode="Topological", show_heatmap=True, heatmap_scale_mode=1,
        heatmap_palette_index=1, weight_mode_index=1, rotation_mode_index=1,
        rotation_field_scores={"H": 1.25, "V": 0.5, "selected": "H"}, machine_angle=90,
        strategy="Best fit", router="A*", heuristic="Manhattan", graph_type="Hannan",
        room_start_mode="Preferred", selected_route_name="Kitchen", preferred_terminal_count=3,
        bend_value=100, bend_min=10, bend_max=200, crossing_value=2, crossing_min=1,
        crossing_max=5, scenario_summary={"routing_frame": {"name": "cover_axes"}},
        fallback_frame_name="world_axes", machine_position=(12, 34), status="Routing OK",
        validation_warnings=("crossing",), elapsed_ms=4.5, total_nodes=99, fps=60,
    )

    view = build_sidebar_view(state)

    assert view.auto_placement.heatmap == "Viridis / Log"
    assert view.auto_placement.placement_weights == "Equal (1.0)"
    assert view.machine.frame == "cover axes"
    assert view.machine.rotation == "Rot: 90° Field H H1.250/V0.500"
    assert view.solver.preferred_terminal_count == 3
    assert view.execution.validation_warnings == ("crossing",)
