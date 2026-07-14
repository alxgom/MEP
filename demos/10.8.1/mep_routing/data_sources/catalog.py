"""Discovery adapter for real dwelling cases exported to SQLite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class DwellingCase:
    project_guid: str
    execution: str
    dwelling_id: str

    @property
    def label(self) -> str:
        return f"{self.execution} / {self.dwelling_id}"


def dwelling_cases_from_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[DwellingCase, ...]:
    """Normalize database catalog rows into stable application cases."""
    cases = {
        DwellingCase(str(row["project_guid"]), str(row["execution"]), str(row["dwelling_id"]))
        for row in rows
    }
    return tuple(sorted(cases, key=lambda case: (case.execution, case.dwelling_id, case.project_guid)))


def discover_dwelling_cases(db_path: Path) -> tuple[DwellingCase, ...]:
    """Read every exported case through dwelling-export's public database API."""
    from dwelling_export.db import connect, list_dwellings

    if not db_path.exists():
        return ()
    with connect(db_path) as connection:
        return dwelling_cases_from_rows(list_dwellings(connection))
