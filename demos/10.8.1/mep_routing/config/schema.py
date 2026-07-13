from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ConfigParameter:
    """One semantic configuration parameter plus migration metadata."""

    key: str
    default: Any
    reason: str
    unit: str | None = None
    description: str = ""
    also_affects: tuple[str, ...] = field(default_factory=tuple)
    legacy_names: tuple[str, ...] = field(default_factory=tuple)


class ConfigCatalog:
    """Lookup catalog that accepts semantic keys and known legacy names."""

    def __init__(self, parameters: Iterable[ConfigParameter]):
        self._parameters: dict[str, ConfigParameter] = {}
        self._aliases: dict[str, str] = {}

        for parameter in parameters:
            if parameter.key in self._parameters:
                raise ValueError(f"duplicate config key: {parameter.key}")
            self._parameters[parameter.key] = parameter
            for legacy_name in parameter.legacy_names:
                existing = self._aliases.get(legacy_name)
                if existing is not None and existing != parameter.key:
                    raise ValueError(
                        f"legacy config name {legacy_name!r} maps to both "
                        f"{existing!r} and {parameter.key!r}"
                    )
                self._aliases[legacy_name] = parameter.key

    def resolve_key(self, key_or_legacy_name: str) -> str:
        return self._aliases.get(key_or_legacy_name, key_or_legacy_name)

    def get_parameter(self, key_or_legacy_name: str) -> ConfigParameter:
        key = self.resolve_key(key_or_legacy_name)
        try:
            return self._parameters[key]
        except KeyError as exc:
            raise KeyError(f"unknown config key: {key_or_legacy_name}") from exc

    def get_default(self, key_or_legacy_name: str) -> Any:
        return self.get_parameter(key_or_legacy_name).default

    def defaults(self) -> dict[str, Any]:
        return {key: parameter.default for key, parameter in self._parameters.items()}

    def legacy_defaults(self) -> dict[str, Any]:
        return {
            legacy_name: self._parameters[key].default
            for legacy_name, key in self._aliases.items()
        }

    def merged_defaults(self, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        values = self.defaults()
        if overrides:
            for key_or_legacy_name, value in overrides.items():
                values[self.resolve_key(key_or_legacy_name)] = value
        return values

    def parameters(self) -> tuple[ConfigParameter, ...]:
        return tuple(self._parameters.values())

