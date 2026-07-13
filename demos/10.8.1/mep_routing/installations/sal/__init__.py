"""Sal ventilation catalog and route policy."""

from .catalog import SAL_OZEO_FLAT_MACHINE
from .routes import KITCHEN_ROUTE_NAME, LARGE_DUCT_ROUTE_NAMES, SHAFT_ROUTE_NAME

__all__ = [
    "KITCHEN_ROUTE_NAME",
    "LARGE_DUCT_ROUTE_NAMES",
    "SAL_OZEO_FLAT_MACHINE",
    "SHAFT_ROUTE_NAME",
]
