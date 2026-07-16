"""Small immutable catalog contracts shared by installation domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterator, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class CatalogEntry(Generic[T]):
    key: str
    label: str
    value: T
    detail: str = ""


@dataclass(frozen=True)
class Catalog(Generic[T]):
    entries: tuple[CatalogEntry[T], ...]
    default_key: str

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("catalog must contain at least one entry")
        keys = tuple(entry.key for entry in self.entries)
        if any(not key for key in keys):
            raise ValueError("catalog entry keys must be non-empty")
        if len(set(keys)) != len(keys):
            raise ValueError("catalog entry keys must be unique")
        if self.default_key not in keys:
            raise ValueError("default key must identify a catalog entry")

    def __iter__(self) -> Iterator[CatalogEntry[T]]:
        return iter(self.entries)

    def entry(self, key: str) -> CatalogEntry[T]:
        entry = next((entry for entry in self.entries if entry.key == key), None)
        if entry is None:
            raise KeyError(key)
        return entry

    def value(self, key: str) -> T:
        return self.entry(key).value

    @property
    def default(self) -> T:
        return self.value(self.default_key)

    @property
    def selectable(self) -> bool:
        return len(self.entries) > 1
