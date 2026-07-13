from mep_routing.ui.sidebar import (
    format_selected_route,
    format_validation_warning,
    wrap_status_lines,
)


def test_sidebar_status_wrapping_preserves_words_and_skips_empty_lines():
    assert wrap_status_lines("routing completed with no warnings", 12) == (
        "routing",
        "completed",
        "with no",
        "warnings",
    )
    assert wrap_status_lines("supercalifragilistic", 10) == ("supercalifragilistic",)


def test_sidebar_warning_and_selection_summaries_keep_dashboard_limits():
    assert format_validation_warning(()) == "Warnings: none"
    assert format_validation_warning(("one", "two", "three")) == "Warnings: one, two +1"
    assert format_validation_warning(("long warning text",), max_length=12) == "Warnings: lo"
    assert format_selected_route("Kitchen extraction route", 3) == "Selected: Kitchen extrac | Prefs: 3"
