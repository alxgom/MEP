from __future__ import annotations

from .defaults_sal import SALUBRIDAD_DEFAULTS


LEGACY_CONSTANT_TO_KEY = {
    legacy_name: SALUBRIDAD_DEFAULTS.resolve_key(legacy_name)
    for legacy_name in SALUBRIDAD_DEFAULTS.legacy_defaults()
}

