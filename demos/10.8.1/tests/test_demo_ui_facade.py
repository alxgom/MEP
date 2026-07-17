from types import SimpleNamespace

from mep_routing.ui.demo_facade import DemoUiFacade
from mep_routing.ui.sidebar import CLI_RETURN_TOGGLE_BOUNDS
from mep_routing.ui.sidebar import solver_dropdown_bounds
from mep_routing.app import WorkbenchState, build_demo_config
from mep_routing.installations.cli import CLI_INSTALLATION
from mep_routing.installations.sal import SAL_INSTALLATION


class _Rect:
    def __init__(self, bounds):
        self.x, self.y, self.width, self.height = bounds

    def collidepoint(self, point):
        x, y = point
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


def test_terminal_tool_is_disabled_outside_sal():
    app = SimpleNamespace(
        installation_header_state=SimpleNamespace(active="cli", enabled={"cli"}),
        routing_controller=SimpleNamespace(installations=SimpleNamespace(
            get=lambda key: SimpleNamespace(capabilities=SimpleNamespace(terminal_tools=False)),
        )),
        preferred_terminal_tool_mode="point",
    )
    facade = DemoUiFacade(app, SimpleNamespace())
    assert facade.handle_terminal_tool_button_click((0, 0)) is None
    assert app.preferred_terminal_tool_mode is None


def test_route_width_uses_real_diameter_or_installation_default():
    app = SimpleNamespace(
        route_real_diameter_width_enabled=True,
        routing_controller=SimpleNamespace(route_diameter=lambda name: 125),
        canvas_ui=SimpleNamespace(viewport=SimpleNamespace(scale=0.1)),
        SAL_INSTALLATION=SimpleNamespace(is_large_route=lambda name: name == "Shaft"),
    )
    facade = DemoUiFacade(app, SimpleNamespace())
    assert facade.get_route_draw_width("Bath") == 12
    app.route_real_diameter_width_enabled = False
    assert facade.get_route_draw_width("Shaft") == 5
    assert facade.get_route_draw_width("Bath") == 3


def test_cli_return_toggle_is_scoped_to_active_cli():
    app = SimpleNamespace(
        installation_header_state=SimpleNamespace(active="cli"),
        workbench_state=SimpleNamespace(
            cli=SimpleNamespace(return_ventilation_enabled=True),
        ),
    )
    facade = DemoUiFacade(app, SimpleNamespace(Rect=_Rect))
    x, y, width, height = CLI_RETURN_TOGGLE_BOUNDS

    assert facade.handle_cli_return_toggle_click((x + width // 2, y + height // 2))
    assert app.workbench_state.cli.return_ventilation_enabled is False

    app.installation_header_state.active = "sal"
    assert not facade.handle_cli_return_toggle_click((x + 1, y + 1))


def test_solver_dropdown_updates_only_active_installation_state():
    workbench_state = WorkbenchState.create()
    app = SimpleNamespace(
        installation_header_state=SimpleNamespace(active="cli"),
        workbench_state=workbench_state,
        CLI_INSTALLATION=CLI_INSTALLATION,
        demo_config=build_demo_config(SAL_INSTALLATION),
    )
    facade = DemoUiFacade(app, SimpleNamespace(Rect=_Rect))
    graph_row = 3
    control, option_bounds = solver_dropdown_bounds(
        graph_row, len(CLI_INSTALLATION.graph_labels),
    )

    assert facade.handle_solver_dropdown_click((control[0] + 1, control[1] + 1))[0]
    handled, changed, field = facade.handle_solver_dropdown_click(
        (option_bounds[1][0] + 1, option_bounds[1][1] + 1),
    )

    assert (handled, changed, field) == (True, True, "graph")
    assert workbench_state.cli.graph_type_idx == 1
