from mep_routing.ui.plots import format_current_value


def test_current_plot_values_use_metric_units_and_precision():
    assert format_current_value(0, 12.345) == "12.3 m"
    assert format_current_value(1, 1234.0) == "1234"
    assert format_current_value(2, 7.0) == "7"
    assert format_current_value(3, 0.456) == "0.46 turns/m"
    assert format_current_value(4, 18.25) == "18.2 ms"
