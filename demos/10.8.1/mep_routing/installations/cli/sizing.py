"""Cli supply-tree demand accumulation and interchangeable duct sizing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


Edge = tuple[int, int]


def canonical_edge(left: int, right: int) -> Edge:
    return (left, right) if left < right else (right, left)


@dataclass(frozen=True)
class CliSizingContext:
    total_airflow_m3h: float
    pressure_loss_mmca: float
    height_mm: int = 150
    width_increment_mm: int = 50


@dataclass(frozen=True)
class CliDuctSize:
    downstream_area_m2: float
    airflow_m3h: float
    width_mm: int
    height_mm: int
    equivalent_diameter_mm: float
    section_m2: float
    nominal_velocity_m_s: float
    actual_velocity_m_s: float


class CliDuctSizingPolicy(Protocol):
    def size(self, downstream_area_m2: float, total_area_m2: float) -> CliDuctSize:
        ...


@dataclass(frozen=True)
class CliGrilleSize:
    room_area_m2: float
    airflow_m3h: float
    width_mm: int
    height_mm: int
    face_velocity_m_s: float


class CliGrilleSizingPolicy(Protocol):
    def size(self, room_area_m2: float, total_area_m2: float) -> CliGrilleSize:
        ...


class CoreSupplyGrilleSizing:
    """Port of ``InfoClima.get_width_reja_impulsion`` and ``_reja``."""

    def __init__(
        self, total_airflow_m3h: float, *, height_mm=150,
        velocities_m_s=(3.0, 3.5, 4.0), width_increment_mm=100,
        maximum_width_mm=1000,
    ):
        self.total_airflow_m3h = float(total_airflow_m3h)
        self.height_mm = int(height_mm)
        self.velocities_m_s = tuple(map(float, velocities_m_s))
        self.width_increment_mm = int(width_increment_mm)
        self.maximum_width_mm = int(maximum_width_mm)

    def size(self, room_area_m2: float, total_area_m2: float) -> CliGrilleSize:
        if self.total_airflow_m3h <= 0 or room_area_m2 < 0 or total_area_m2 <= 0:
            raise ValueError("grille airflow and sizing areas must be positive")
        airflow = self.total_airflow_m3h * room_area_m2 / total_area_m2
        height_m = self.height_mm / 1000.0
        width_mm = self.maximum_width_mm
        for velocity in self.velocities_m_s:
            # Core doubles the free area requirement to model a 50% effective area.
            required_area = airflow / (3600.0 * velocity) / 0.5
            raw_width_mm = required_area / height_m * 1000.0
            candidate = int(
                math.ceil(raw_width_mm / self.width_increment_mm)
                * self.width_increment_mm
            )
            if candidate <= self.maximum_width_mm:
                width_mm = candidate
                break
        face_velocity = (airflow * 2.0) / (
            (width_mm / 1000.0) * height_m * 3600.0
        )
        return CliGrilleSize(
            room_area_m2, airflow, width_mm, self.height_mm, face_velocity,
        )


class CoreConstantFrictionSizing:
    """Port of ``InfoClima.get_dimensiones_conducto`` and ``_conducto``."""

    def __init__(self, context: CliSizingContext):
        if context.total_airflow_m3h <= 0 or context.pressure_loss_mmca <= 0:
            raise ValueError("airflow and pressure loss must be positive")
        if context.height_mm <= 0 or context.width_increment_mm <= 0:
            raise ValueError("duct height and width increment must be positive")
        self.context = context

    def size(self, downstream_area_m2: float, total_area_m2: float) -> CliDuctSize:
        if downstream_area_m2 < 0 or total_area_m2 <= 0:
            raise ValueError("sizing areas must be nonnegative with a positive total")
        airflow = round(
            self.context.total_airflow_m3h * downstream_area_m2 / total_area_m2, 2,
        )
        if airflow <= 0:
            return CliDuctSize(downstream_area_m2, airflow, 0, self.context.height_mm, 0, 0, 0, 0)
        flow_m3_s = airflow / 3600.0
        diameter_m = (
            2.09e-3 * (flow_m3_s ** 1.82) / self.context.pressure_loss_mmca
        ) ** (1.0 / 4.86)
        section_m2 = math.pi * (diameter_m / 2.0) ** 2
        nominal_velocity = flow_m3_s / section_m2
        height_m = self.context.height_mm / 1000.0
        raw_width_mm = section_m2 / height_m * 1000.0
        increment = self.context.width_increment_mm
        width_mm = max(self.context.height_mm, int(math.ceil(raw_width_mm / increment) * increment))
        actual_velocity = flow_m3_s / ((width_mm / 1000.0) * height_m)
        return CliDuctSize(
            downstream_area_m2, airflow, width_mm, self.context.height_mm,
            diameter_m * 1000.0, section_m2, nominal_velocity, actual_velocity,
        )


def size_tree_edges(
    tree_edges, root_node: int, terminal_area_m2_by_node: dict[int, float],
    policy: CliDuctSizingPolicy,
) -> dict[Edge, CliDuctSize]:
    """Orient an undirected tree at the machine and size edges in post-order."""
    adjacency: dict[int, list[int]] = {}
    for raw_left, raw_right in tree_edges:
        left, right = int(raw_left), int(raw_right)
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
    if root_node not in adjacency:
        return {}
    parent = {int(root_node): None}
    order = [int(root_node)]
    for node in order:
        for neighbour in sorted(adjacency.get(node, ())):
            if neighbour in parent:
                continue
            parent[neighbour] = node
            order.append(neighbour)
    total_area = float(sum(terminal_area_m2_by_node.values()))
    if total_area <= 0:
        return {}
    downstream = {node: float(terminal_area_m2_by_node.get(node, 0.0)) for node in order}
    result = {}
    for node in reversed(order[1:]):
        upstream = parent[node]
        result[canonical_edge(upstream, node)] = policy.size(downstream[node], total_area)
        downstream[upstream] += downstream[node]
    return result
