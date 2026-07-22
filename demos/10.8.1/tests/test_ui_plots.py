from mep_routing.ui.plots import (
    format_current_value, format_minimum_value,
    kpi_scope_action, kpi_scope_dropdown_bounds,
)


def test_current_plot_values_use_metric_units_and_precision():
    assert format_current_value(0, 12.345) == "12.3 m"
    assert format_current_value(1, 1234.0) == "1234"
    assert format_current_value(2, 7.0) == "7"
    assert format_current_value(3, 0.456) == "0.46 turns/m"
    assert format_current_value(4, 18.25) == "18.2 ms"


def test_minimum_plot_values_keep_metric_units_without_decimal_noise():
    assert format_minimum_value(0, 4.004) == "Best 4.00 m"
    assert format_minimum_value(1, 16699.2) == "Best 16699"
    assert format_minimum_value(2, 2.0) == "Best 2"
    assert format_minimum_value(3, 0.204) == "Best 0.20 turns/m"
    assert format_minimum_value(4, 110.44) == "Best 110.4 ms"


def test_kpi_scope_dropdown_toggles_and_selects_options():
    options = (("all", "All installations", (1, 2, 3)), ("sal", "Sal", (4, 5, 6)))
    control, option_bounds = kpi_scope_dropdown_bounds(1700, 280, len(options))

    assert kpi_scope_action(
        (control[0] + 1, control[1] + 1), 1700, 280, options, False,
    ) == ("toggle", None)
    sal_bounds = option_bounds[1]
    assert kpi_scope_action(
        (sal_bounds[0] + 1, sal_bounds[1] + 1), 1700, 280, options, True,
    ) == ("choose", "sal")
    assert kpi_scope_action((0, 0), 1700, 280, options, False) is None
    assert kpi_scope_action((0, 0), 1700, 280, options, True) == ("dismiss", None)
