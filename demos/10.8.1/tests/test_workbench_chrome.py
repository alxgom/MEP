from types import SimpleNamespace

from mep_routing.ui.workbench_chrome import WorkbenchChromeRuntime


def _chrome():
    viewport = SimpleNamespace(canvas_left=0, canvas_top=0, canvas_rect=(0, 0, 1, 1))
    return WorkbenchChromeRuntime(viewport, SimpleNamespace(), 280, {})


def test_transient_message_expires_after_deadline():
    chrome = _chrome()
    chrome.set_transient_message("pending", now_ms=100, duration_ms=20)
    assert chrome.transient_message_at(120) == "pending"
    assert chrome.transient_message_at(121) is None


def test_help_toggle_closes_active_card():
    chrome = _chrome()
    chrome.toggle_help("solver")
    assert chrome.help_popup_card == "solver"
    chrome.toggle_help("solver")
    assert chrome.help_popup_card is None

