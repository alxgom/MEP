from __future__ import annotations

from dataclasses import dataclass, field

from .routes import LARGE_DUCT_ROUTE_NAMES


@dataclass(frozen=True)
class MachineSpec:
    """Machine and connector dimensions in integer millimetres."""

    name: str
    body_width_mm: int
    body_height_mm: int
    overall_width_mm: int
    small_duct_diameter_mm: int
    large_duct_diameter_mm: int
    small_pin_stub_length_mm: int
    large_pin_stub_length_mm: int
    large_route_names: frozenset[str] = field(default_factory=lambda: LARGE_DUCT_ROUTE_NAMES)

    def route_diameter_mm(self, route_name: str) -> int:
        if route_name in self.large_route_names:
            return self.large_duct_diameter_mm
        return self.small_duct_diameter_mm

    def pin_stub_length_mm(self, pin_name: str) -> int:
        if pin_name in ("left_mid", "right_mid"):
            return self.large_pin_stub_length_mm
        return self.small_pin_stub_length_mm


SAL_OZEO_FLAT_MACHINE = MachineSpec(
    name="S&P Ozeo Flat",
    body_width_mm=410,
    body_height_mm=460,
    overall_width_mm=511,
    small_duct_diameter_mm=90,
    large_duct_diameter_mm=125,
    small_pin_stub_length_mm=100,
    large_pin_stub_length_mm=250,
)
