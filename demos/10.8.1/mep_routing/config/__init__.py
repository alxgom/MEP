"""Configuration catalog and defaults for the Demo 10.8.1 refactor."""

from .defaults_sal import SALUBRIDAD_DEFAULTS, build_sal_catalog
from .schema import ConfigCatalog, ConfigParameter

__all__ = [
    "ConfigCatalog",
    "ConfigParameter",
    "SALUBRIDAD_DEFAULTS",
    "build_sal_catalog",
]

