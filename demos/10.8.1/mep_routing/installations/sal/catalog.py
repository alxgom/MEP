"""Machine catalog for the current Sal ventilation installation."""

from mep_routing.domain import MachineSpec

from .routes import LARGE_DUCT_ROUTE_NAMES


SAL_OZEO_FLAT_MACHINE = MachineSpec(
    name="S&P Ozeo Flat",
    # AzureFile family: Sal_EquipoVentilacion_Soler&Palau_Ozeo_Flat_Ozeo_Flat_Auto_2V.json
    body_width_mm=459,
    body_height_mm=459,
    overall_width_mm=512,
    small_duct_diameter_mm=80,
    large_duct_diameter_mm=125,
    small_pin_stub_length_mm=100,
    large_pin_stub_length_mm=250,
    large_route_names=LARGE_DUCT_ROUTE_NAMES,
    installation_height_mm=197.5035496888521,
    installation_clearance_mm=20.0,
)
