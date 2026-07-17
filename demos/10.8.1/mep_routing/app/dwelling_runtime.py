"""Dwelling source preparation and activation for the interactive demo."""

from __future__ import annotations

from mep_routing.app.dwelling_application import create_active_dwelling, load_real_dwelling
from mep_routing.app.graph_build_session import GraphBuildSession
from mep_routing.data_sources import (
    build_wall_polygons,
    build_synthetic_dwelling,
    choose_initial_machine_position,
    derive_room_boundary_walls,
    prepare_real_dwelling,
    prepare_synthetic_dwelling,
)
from mep_routing.placement import insufficient_machine_clearance_regions


class DwellingRuntime:
    """Own source fallback and the ordered activation of one prepared dwelling."""

    def __init__(self, app):
        self.app = app

    def generate_synthetic(self):
        a = self.app
        scenario = build_synthetic_dwelling(
            a.generative_layout,
            a.generative_layout.Room,
            a.get_representative_point,
            scale_to_mm=a.demo_config.scale_to_mm,
            wall_thickness_mm=a.demo_config.wall_thickness_mm,
        )
        self.apply_prepared(prepare_synthetic_dwelling(scenario), auto_place=False)

    def load_real(self):
        a = self.app
        if not a.REAL_DWELLING_SCENARIOS:
            raise RuntimeError("Dwelling export database contains no cases.")
        case = a.REAL_DWELLING_SCENARIOS[a.real_scenario_idx % len(a.REAL_DWELLING_SCENARIOS)]
        return load_real_dwelling(
            loader=a.load_dwelling_scenario,
            summarizer=a.scenario_summary,
            db_path=a.demo_config.real_dwelling_db,
            case=case,
            frame_name=a.demo_config.routing_frame_options[
                a.routing_frame_idx % len(a.demo_config.routing_frame_options)
            ],
            preferred_shaft_installation=a.demo_config.preferred_shaft_installation,
        )

    def generate(self):
        a = self.app
        a.ui_facade.invalidate_room_start_node_cache()
        if a.demo_config.dwelling_source_modes[a.dwelling_source_idx] == "Random Synthetic":
            self.generate_synthetic()
            return
        try:
            scenario, label, summary = self.load_real()
        except Exception as error:
            print(f"Real dwelling load failed, falling back to synthetic: {error}")
            self.generate_synthetic()
            return
        prepared = prepare_real_dwelling(
            scenario,
            wall_thickness_mm=a.demo_config.wall_thickness_mm,
            label=label,
            summary=summary,
            derive_walls=derive_room_boundary_walls,
            build_wall_polygons=build_wall_polygons,
            choose_machine_position=lambda terminals, shaft: choose_initial_machine_position(
                terminals, shaft, a.get_representative_point,
            ),
            build_core_entry_specs=a._build_core_shaft_entry_specs,
            build_installation_inputs=a._build_installation_inputs,
        )
        self.apply_prepared(prepared, auto_place=True)

    def apply_prepared(self, prepared, *, auto_place):
        """Activate sessions in dependency order and publish UI compatibility state."""
        a = self.app
        dwelling = create_active_dwelling(
            prepared, a.demo_config.common_graph, a.SAL_GRAPH_PROFILE,
            a.demo_config.terminal_runtime,
        )
        state = a.workbench_state.dwelling
        state.active = dwelling
        state.workspace = dwelling.workspace
        state.terminal_runtime = dwelling.terminals
        a.terminal_ui.bind(dwelling)
        state.graph_build_session = GraphBuildSession(
            state.workspace,
            state.terminal_runtime,
            a.ui_facade.invalidate_room_start_node_cache,
            a.ui_facade.invalidate_terminal_validity_cache,
        )

        aliases = {
            "rooms": dwelling.rooms,
            "columns": dwelling.columns,
            "shafts": dwelling.shafts,
            "covers": dwelling.covers,
            "doors": dwelling.doors,
            "walls": dwelling.walls,
            "wall_polys": dwelling.wall_polygons,
            "routing_region_base": dwelling.routing_region,
            "shaft_extraction": dwelling.shaft_extraction,
            "terminals": dwelling.terminal_points,
            "wet_room_names": dwelling.wet_room_names,
            "wet_room_outer_accents": prepared.wet_room_outer_accents,
            "current_scenario_label": prepared.label,
            "current_scenario_summary": prepared.summary,
            "shaft_core_entry_specs": dwelling.shaft_core_entry_specs,
            "_bnd_segs": None,
        }
        for name, value in aliases.items():
            setattr(a, name, value)

        sal_pose = a.workbench_state.installation("sal").pose
        cli_pose = a.workbench_state.installation("cli").pose
        coc_pose = a.workbench_state.installation("coc").pose
        sal_pose.set(prepared.machine_position, 0)
        cli_machine = a.CLI_INSTALLATION.machine_catalog.value(
            a.installation_ui_session.selected_keys["cli"],
        )
        cli_placement = a.cli_runtime.place_machine(prepared, cli_machine)
        cli_pose.set(cli_placement.center, cli_placement.angle_degrees)
        coc_payload = (prepared.installation_inputs or {}).get("coc") or {}
        coc_source = (coc_payload.get("appliance") or {}).get("exhaust_xy")
        coc_pose.set(tuple(coc_source) if coc_source else (0.0, 0.0), 0)
        state.machine_vertical_clearance_blocks = insufficient_machine_clearance_regions(
            dwelling.rooms,
            a.SAL_INSTALLATION.default_machine.installation_height_mm,
            a.SAL_INSTALLATION.default_machine.installation_clearance_mm,
        )
        a.placement_session.bind(
            state.workspace, dwelling.rooms, dwelling.terminal_points,
            dwelling.wet_room_names, dwelling.routing_region, dwelling.shaft_extraction,
            state.machine_vertical_clearance_blocks,
            (*dwelling.walls, *dwelling.wall_polygons), dwelling.columns, dwelling.shafts,
        )
        state.graph_build_session.build_base_regular()
        if auto_place:
            a.routing_controller.auto_place()
        pins = a.routing_controller.machine_pins(*sal_pose.as_tuple())
        a.routing_controller.build_grid(machine_pins=pins)
