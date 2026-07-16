"""Cli domain foundations; no Cli routing runtime is enabled yet."""

from .catalog import CLI_MACHINE_CATALOG, CLI_PEAD_MACHINES, CliConnector, CliMachine
from .definition import CLI_INSTALLATION, CliInstallationDefinition

__all__ = [
    "CLI_INSTALLATION", "CLI_MACHINE_CATALOG", "CLI_PEAD_MACHINES",
    "CliConnector", "CliInstallationDefinition", "CliMachine",
]
