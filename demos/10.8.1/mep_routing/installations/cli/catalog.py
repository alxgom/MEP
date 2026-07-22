"""Indoor Clima machine catalog ported from Demo 10.8_cli metadata."""

from __future__ import annotations

from dataclasses import dataclass

from mep_routing.domain import Catalog, CatalogEntry


@dataclass(frozen=True)
class CliConnector:
    key: str
    description: str
    offset_mm: tuple[int, int, int]
    normal_xy: tuple[int, int]
    rectangular_size_mm: tuple[int, int] | None = None
    radius_mm: float | None = None


@dataclass(frozen=True)
class CliMachine:
    family: str
    type_name: str
    body_size_mm: tuple[int, int, int]
    insertion_mm: tuple[int, int, int]
    connectors: tuple[CliConnector, ...]
    airflow_m3h: float | None = None
    pressure_loss_mmca: float | None = None

    def connector(self, key: str) -> CliConnector:
        return next(connector for connector in self.connectors if connector.key == key)


def _pead(
    type_name, width, insertion_x, air_width, freon1_x, freon2_x, r1, r2,
    airflow_m3h=None, pressure_loss_mmca=None,
):
    return CliMachine(
        family="Cli_UnidadInterior_Mitsubishi_PEAD",
        type_name=type_name,
        body_size_mm=(width, 700, 250),
        insertion_mm=(insertion_x, 366, 250),
        connectors=(
            CliConnector("air_out", "Supply Air1", (0, 334, -112), (0, 1), (air_width, 178)),
            CliConnector("air_in", "Supply Air2", (0, -366, -125), (0, -1), (air_width, 178)),
            CliConnector("freon1", "Freon1", (freon1_x, 198, -122), (-1, 0), radius_mm=r1),
            CliConnector("freon2", "Freon2", (freon2_x, 131, -155), (-1, 0), radius_mm=r2),
        ),
        airflow_m3h=airflow_m3h,
        pressure_loss_mmca=pressure_loss_mmca,
    )


CLI_PEAD_MACHINES = (
    _pead("M35JA", 1028, 528, 860, -507, -528, 3.18, 6.35),
    _pead("M50JA", 1028, 528, 860, -507, -528, 3.18, 6.35),
    _pead("M60JA", 1228, 628, 1060, -607, -628, 4.76, 7.95),
    # Airflow/loss pairs are verified routing-core project inputs. Other
    # catalog models remain explicitly unsourced instead of inferring values.
    _pead("M71JA", 1228, 628, 1060, -607, -628, 4.76, 7.95, 1260.0, 0.06),
    _pead("M100JA", 1528, 778, 1360, -757, -778, 4.76, 7.95, 1740.0, 0.047),
    _pead("M125JA", 1528, 778, 1360, -757, -778, 4.76, 7.95),
    _pead("M140JA", 1728, 878, 1560, -857, -878, 4.76, 7.95),
)

CLI_MACHINE_CATALOG = Catalog(
    entries=tuple(
        CatalogEntry(
            key=f"pead-{machine.type_name.lower()}",
            label=f"PEAD {machine.type_name}",
            value=machine,
            detail=" x ".join(map(str, machine.body_size_mm)) + " mm",
        )
        for machine in CLI_PEAD_MACHINES
    ),
    default_key="pead-m71ja",
)
