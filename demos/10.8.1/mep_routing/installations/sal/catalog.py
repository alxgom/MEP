"""Machine catalog for the current Sal ventilation installation."""

from mep_routing.domain import MachineSpec

from .routes import LARGE_DUCT_ROUTE_NAMES


SAL_OZEO_FLAT_MACHINE = MachineSpec(
    name="S&P Ozeo Flat",
    body_width_mm=410,
    body_height_mm=460,
    overall_width_mm=511,
    small_duct_diameter_mm=90,
    large_duct_diameter_mm=125,
    small_pin_stub_length_mm=100,
    large_pin_stub_length_mm=250,
    large_route_names=LARGE_DUCT_ROUTE_NAMES,
)
