from types import SimpleNamespace

from mep_routing.ui.demo_facade import DemoUiFacade


def test_terminal_tool_is_disabled_outside_sal():
    app = SimpleNamespace(
        installation_header_state=SimpleNamespace(active="cli"),
        preferred_terminal_tool_mode="point",
    )
    facade = DemoUiFacade(app, SimpleNamespace())
    assert facade.handle_terminal_tool_button_click((0, 0)) is None
    assert app.preferred_terminal_tool_mode is None


def test_route_width_uses_real_diameter_or_installation_default():
    app = SimpleNamespace(
        route_real_diameter_width_enabled=True,
        get_route_diameter=lambda name: 125,
        canvas_ui=SimpleNamespace(viewport=SimpleNamespace(scale=0.1)),
        SAL_INSTALLATION=SimpleNamespace(is_large_route=lambda name: name == "Shaft"),
    )
    facade = DemoUiFacade(app, SimpleNamespace())
    assert facade.get_route_draw_width("Bath") == 12
    app.route_real_diameter_width_enabled = False
    assert facade.get_route_draw_width("Shaft") == 5
    assert facade.get_route_draw_width("Bath") == 3
