"""Concrete application surface consumed by the Pygame workbench."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkbenchSmokeResult:
    """Small, stable result returned by the non-interactive startup check."""

    dwelling_label: str
    route_count: int
    status: str
    elapsed_ms: float
    total_nodes: int


def run_workbench_smoke(application):
    """Generate one dwelling and route it without starting the Pygame loop."""

    application.dwelling_runtime.generate()
    routes, status, elapsed_ms, total_nodes = (
        application.routing_controller.solve_installations()
    )
    active = application.workbench_state.dwelling.active
    if active is None:
        raise RuntimeError("Workbench smoke test did not create an active dwelling")
    return WorkbenchSmokeResult(
        dwelling_label=application.current_scenario_label,
        route_count=len(routes),
        status=status,
        elapsed_ms=elapsed_ms,
        total_nodes=total_nodes,
    )


class DemoWorkbenchApplication:
    """Own application controllers while legacy composition data is migrated.

    The fallback namespace contains values only.  Controller callbacks are
    deliberately not flattened onto it, so UI adapters depend on this concrete
    object and its named controller boundaries.
    """

    _OWN_FIELDS = {
        "_composition", "routing_controller", "dwelling_runtime", "ui_facade",
        "commands", "observation", "chrome",
    }

    def __init__(self, composition):
        object.__setattr__(self, "_composition", composition)
        for name in self._OWN_FIELDS - {"_composition"}:
            object.__setattr__(self, name, None)

    def bind(self, *, routing, dwelling, ui, commands, observation, chrome):
        object.__setattr__(self, "routing_controller", routing)
        object.__setattr__(self, "dwelling_runtime", dwelling)
        object.__setattr__(self, "ui_facade", ui)
        object.__setattr__(self, "commands", commands)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "chrome", chrome)
        return self

    def __getattr__(self, name):
        return getattr(self._composition, name)

    def __setattr__(self, name, value):
        if name in self._OWN_FIELDS:
            object.__setattr__(self, name, value)
        else:
            setattr(self._composition, name, value)

    @property
    def observation_runtime(self):
        return self.observation.runtime

    @property
    def routing_history(self):
        return self.observation.history

    @property
    def solution_log_session(self):
        return self.observation.logs

    def get_active_machine_pose(self):
        active = self.installation_header_state.active
        return self.workbench_state.installation(active).pose.as_tuple()

    def move_active_machine(self, position):
        active = self.installation_header_state.active
        self.workbench_state.installation(active).pose.set(position)

    def rotate_active_machine(self, delta_degrees):
        active = self.installation_header_state.active
        return self.workbench_state.installation(active).pose.rotate(delta_degrees)

    def get_active_machine_pins(self, cx, cy, angle_degrees):
        if self.installation_header_state.active == "coc":
            state = self.workbench_state.installation("coc")
            runtime = self.routing_controller.installations.get("coc")
            return dict(runtime.machine_pins(state))
        if self.installation_header_state.active == "cli":
            machine_key = self.installation_ui_session.selected_keys["cli"]
            machine = self.CLI_INSTALLATION.machine_catalog.value(machine_key)
            return self.cli_machine_pins(machine, (cx, cy), angle_degrees)
        return self.routing_controller.machine_pins(cx, cy, angle_degrees)

    @staticmethod
    def get_representative_point(polygon):
        centroid = polygon.centroid
        point = centroid if polygon.contains(centroid) else polygon.representative_point()
        return round(point.x), round(point.y)

    def _build_core_shaft_entry_specs(self, scenario):
        return self._source_core_shaft_entry_specs(
            scenario, summarize=self.scenario_summary,
            scale_to_mm=self.demo_config.scale_to_mm,
            match_tolerance_mm=self.demo_config.core_epsilon_grid_mm * 0.75,
            installation="Sal",
        )

    def _build_installation_inputs(self, scenario):
        appliances = tuple(getattr(scenario, "coc_appliances", ()) or ())
        appliance = next(
            (item for item in appliances if item.get("selected")),
            appliances[0] if appliances else None,
        )
        shaft_entries = self._source_core_shaft_entry_specs(
            scenario, summarize=self.scenario_summary,
            scale_to_mm=self.demo_config.scale_to_mm,
            match_tolerance_mm=self.demo_config.core_epsilon_grid_mm * 0.75,
            installation="Coc",
        )
        shaft = None
        if shaft_entries:
            x, y = shaft_entries[0]["centroid"]
            def distance_to_bounds(item):
                min_x, min_y, max_x, max_y = item.bounds
                dx = max(min_x - x, 0.0, x - max_x)
                dy = max(min_y - y, 0.0, y - max_y)
                return dx * dx + dy * dy
            shaft = next(
                (item for item in scenario.shafts
                 if item.bounds[0] <= x <= item.bounds[2]
                 and item.bounds[1] <= y <= item.bounds[3]),
                min(scenario.shafts, key=distance_to_bounds, default=None),
            )
        return {"coc": {
            "appliance": appliance,
            "duct": dict(getattr(scenario, "coc_duct", {}) or {}),
            "shaft_entries": tuple(shaft_entries),
            "shaft": shaft,
        }}

    def count_segment_crossings(self, routes):
        return self._count_segment_crossings(routes)

    def count_solution_turns(self, routes):
        return self._count_solution_turns(routes)

    def find_room_route_at_point(self, world_point, route_names):
        return self.terminal_ui.find_room_route(
            world_point, route_names, self.shaft_extraction,
        )

    def draw_wet_room_outer_accents(self, screen):
        return self._draw_wet_room_outer_accents(
            screen, self.wet_room_outer_accents, self.to_screen,
            self._iter_polygons, self.theme.wet_room_accent,
        )

    def draw_outlined_text(self, screen, font, text, position, color, outline_color=None):
        return self._draw_outlined_text(
            screen, font, text, position, color,
            self.theme.plan_label_halo if outline_color is None else outline_color,
        )

    def draw_terminal_area_drag(self, screen, start_world, end_world):
        return self._draw_terminal_area_drag(
            screen, start_world, end_world, self.to_screen,
        )

    def apply_preferred_terminal_point(self, world_point, remove=False):
        if not self.active_capabilities(self).terminal_tools:
            return False, None
        return self.terminal_ui.apply_point_preference(world_point, remove=remove)

    def apply_preferred_terminal_area(self, start_world, end_world, remove=False):
        if not self.active_capabilities(self).terminal_tools:
            return False, None
        return self.terminal_ui.apply_area_preference(start_world, end_world, remove=remove)
