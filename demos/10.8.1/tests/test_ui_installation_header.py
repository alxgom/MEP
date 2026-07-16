from mep_routing.ui.installation_header import (
    InstallationHeaderState,
    InstallationOption,
    activate_installation,
    installation_header_hit,
    installation_header_layout,
    move_installation,
    toggle_installation,
)


OPTIONS = (
    InstallationOption("sal", "Sal", True),
    InstallationOption("cli", "Cli"),
    InstallationOption("san", "San"),
    InstallationOption("coc", "Coc"),
)
STATE = InstallationHeaderState(("sal", "cli", "san", "coc"), frozenset({"sal"}), "sal")


def test_header_keeps_active_and_enabled_as_separate_state():
    disabled, changed = toggle_installation(STATE, "sal", OPTIONS)
    assert changed and disabled.active == "sal" and not disabled.enabled


def test_unavailable_installation_cannot_be_enabled_or_activated():
    assert toggle_installation(STATE, "cli", OPTIONS) == (STATE, False)
    assert activate_installation(STATE, "cli", OPTIONS) == (STATE, False)


def test_installations_can_be_reordered_independently_of_availability():
    reordered = move_installation(STATE, "coc", 1)
    assert reordered.order == ("sal", "coc", "cli", "san")
    assert reordered.enabled == STATE.enabled and reordered.active == "sal"


def test_header_hit_prioritizes_switch_over_pill_body():
    layout = installation_header_layout(320, 4, STATE.order)
    switch = layout.switch_bounds["sal"]
    pill = layout.pill_bounds["sal"]
    assert installation_header_hit(layout, (switch[0] + 2, switch[1] + 2)) == ("switch", "sal")
    assert installation_header_hit(layout, (pill[0] + pill[2] - 4, pill[1] + 4)) == ("pill", "sal")
