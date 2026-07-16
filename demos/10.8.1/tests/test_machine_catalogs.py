import pytest

from mep_routing.domain import Catalog, CatalogEntry, MachineSpec
from mep_routing.graphs import GraphMode
from mep_routing.installations.cli import CLI_INSTALLATION, CLI_MACHINE_CATALOG, CliMachine
from mep_routing.installations.sal import SAL_INSTALLATION, SAL_MACHINE_CATALOG, SAL_OZEO_FLAT_MACHINE


def test_catalog_validates_keys_and_resolves_default():
    catalog = Catalog((CatalogEntry("one", "One", 1),), "one")
    assert catalog.default == 1
    assert catalog.value("one") == 1
    assert not catalog.selectable

    with pytest.raises(ValueError, match="unique"):
        Catalog((CatalogEntry("one", "One", 1), CatalogEntry("one", "Again", 2)), "one")
    with pytest.raises(ValueError, match="default"):
        Catalog((CatalogEntry("one", "One", 1),), "missing")


def test_sal_exposes_its_existing_machine_as_a_one_entry_catalog():
    assert SAL_MACHINE_CATALOG.default is SAL_OZEO_FLAT_MACHINE
    assert SAL_INSTALLATION.machine_catalog is SAL_MACHINE_CATALOG
    assert SAL_INSTALLATION.machines == (SAL_MACHINE_CATALOG.default,)
    assert isinstance(SAL_MACHINE_CATALOG.default, MachineSpec)
    assert not SAL_MACHINE_CATALOG.selectable


def test_cli_catalog_preserves_the_seven_pead_variants_and_default():
    assert tuple(entry.label for entry in CLI_MACHINE_CATALOG) == (
        "PEAD M35JA", "PEAD M50JA", "PEAD M60JA", "PEAD M71JA",
        "PEAD M100JA", "PEAD M125JA", "PEAD M140JA",
    )
    assert CLI_MACHINE_CATALOG.default.type_name == "M71JA"
    assert CLI_MACHINE_CATALOG.selectable
    assert all(isinstance(entry.value, CliMachine) for entry in CLI_MACHINE_CATALOG)
    assert not isinstance(CLI_MACHINE_CATALOG.default, MachineSpec)


def test_cli_machine_keeps_exported_connector_geometry():
    m140 = CLI_MACHINE_CATALOG.value("pead-m140ja")
    assert m140.body_size_mm == (1728, 700, 250)
    assert m140.insertion_mm == (878, 366, 250)
    assert m140.connector("air_out").offset_mm == (0, 334, -112)
    assert m140.connector("air_out").rectangular_size_mm == (1560, 178)
    assert m140.connector("freon2").offset_mm == (-878, 131, -155)
    assert m140.connector("freon2").radius_mm == 7.95


def test_cli_definition_composes_available_interactive_capabilities():
    assert CLI_INSTALLATION.machine_catalog is CLI_MACHINE_CATALOG
    assert CLI_INSTALLATION.supply_graph_mode is GraphMode.CORE_SIMPLE
    assert CLI_INSTALLATION.default_machine.type_name == "M71JA"
    assert CLI_INSTALLATION.interactive_runtime_available is True
