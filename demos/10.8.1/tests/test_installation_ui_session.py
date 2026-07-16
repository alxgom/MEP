from mep_routing.app import InstallationMachineUi, InstallationUiSession
from mep_routing.installations.cli import CLI_INSTALLATION
from mep_routing.installations.sal import SAL_INSTALLATION


def _session():
    return InstallationUiSession.create({
        "sal": InstallationMachineUi(SAL_INSTALLATION.machine_catalog, True),
        "cli": InstallationMachineUi(CLI_INSTALLATION.machine_catalog, False),
    })


def test_machine_keys_and_dropdown_state_are_scoped_per_installation():
    session = _session()
    assert len(session.options("sal")) == 1
    assert len(session.options("cli")) == 7

    transition = session.toggle_dropdown("cli").choose("cli", "pead-m140ja")

    assert transition.changed
    assert not transition.runtime_invalidation_required
    assert transition.session.selected_keys["cli"] == "pead-m140ja"
    assert transition.session.selected_keys["sal"] == SAL_INSTALLATION.machine_catalog.default_key
    assert not transition.session.dropdown_state("cli").open


def test_runtime_backed_machine_change_exposes_future_invalidation_contract():
    session = _session()
    runtime_session = InstallationUiSession.create({
        "cli": InstallationMachineUi(CLI_INSTALLATION.machine_catalog, True),
    })

    unchanged = session.choose("sal", SAL_INSTALLATION.machine_catalog.default_key)
    changed = runtime_session.choose("cli", "pead-m140ja")

    assert not unchanged.changed and not unchanged.runtime_invalidation_required
    assert changed.changed and changed.runtime_invalidation_required
