import pytest

from mep_routing.ui.dropdown import DropdownState, dropdown_bounds, dropdown_transition


def test_dropdown_layout_stacks_options_below_control():
    control, options = dropdown_bounds(10, 20, 100, 2)
    assert control == (10, 20, 100, 24)
    assert options == ((10, 46, 100, 24), (10, 70, 100, 24))


def test_dropdown_transition_reports_only_real_selection_changes():
    state = DropdownState("a")
    opened = dropdown_transition(state, "toggle")
    assert opened.state == DropdownState("a", True)
    same = dropdown_transition(opened.state, "choose", "a")
    assert same.state == DropdownState("a", False)
    assert not same.changed
    changed = dropdown_transition(opened.state, "choose", "b")
    assert changed.state == DropdownState("b", False)
    assert changed.changed


def test_dropdown_transition_rejects_invalid_commands():
    with pytest.raises(ValueError, match="requires"):
        dropdown_transition(DropdownState("a"), "choose")
    with pytest.raises(ValueError, match="unknown"):
        dropdown_transition(DropdownState("a"), "explode")
