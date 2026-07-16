"""Application state for installation-scoped machine selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from mep_routing.domain import Catalog
from mep_routing.ui.dropdown import DropdownOption, DropdownState, dropdown_transition


@dataclass(frozen=True)
class InstallationMachineUi:
    catalog: Catalog
    runtime_available: bool


@dataclass(frozen=True)
class MachineSelectionTransition:
    session: "InstallationUiSession"
    changed: bool
    runtime_invalidation_required: bool


@dataclass(frozen=True)
class InstallationUiSession:
    installations: Mapping[str, InstallationMachineUi]
    selected_keys: Mapping[str, str]
    dropdown_open_for: str | None = None

    @classmethod
    def create(cls, installations: Mapping[str, InstallationMachineUi]):
        return cls(
            dict(installations),
            {key: item.catalog.default_key for key, item in installations.items()},
        )

    def options(self, installation_key: str) -> tuple[DropdownOption, ...]:
        catalog = self.installations[installation_key].catalog
        return tuple(DropdownOption(entry.key, entry.label, entry.detail) for entry in catalog)

    def dropdown_state(self, installation_key: str) -> DropdownState:
        return DropdownState(
            self.selected_keys[installation_key],
            self.dropdown_open_for == installation_key,
        )

    def toggle_dropdown(self, installation_key: str) -> "InstallationUiSession":
        if not self.installations[installation_key].catalog.selectable:
            return self
        open_for = None if self.dropdown_open_for == installation_key else installation_key
        return replace(self, dropdown_open_for=open_for)

    def dismiss_dropdown(self) -> "InstallationUiSession":
        return replace(self, dropdown_open_for=None)

    def choose(self, installation_key: str, machine_key: str) -> MachineSelectionTransition:
        catalog = self.installations[installation_key].catalog
        catalog.entry(machine_key)
        transition = dropdown_transition(self.dropdown_state(installation_key), "choose", machine_key)
        selected = dict(self.selected_keys)
        selected[installation_key] = transition.state.selected_key
        session = replace(self, selected_keys=selected, dropdown_open_for=None)
        return MachineSelectionTransition(
            session, transition.changed,
            transition.changed and self.installations[installation_key].runtime_available,
        )
