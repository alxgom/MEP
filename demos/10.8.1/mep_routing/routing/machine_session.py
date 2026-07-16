"""Live machine geometry at the routing-graph boundary."""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Point, Polygon

from mep_routing.domain import machine_pins, port_access_specs


@dataclass(frozen=True)
class MachineRoutingSession:
    """Snapshot machine placement and adapt it to a live routing workspace."""

    spec: object
    center: tuple[float, float]
    angle: float
    routing_region: object
    columns: tuple
    workspace: object
    graph_type: int
    update_dynamic_env: object
    build_grid: object
    blocked_vertical_regions: tuple = ()

    @property
    def pins(self):
        return machine_pins(self.spec, self.center[0], self.center[1], self.angle)

    @staticmethod
    def polygon(pins):
        return Polygon([pins["c_tl"], pins["c_tr"], pins["c_br"], pins["c_bl"]])

    def preflight_error(self):
        machine_polygon = self.polygon(self.pins)
        if not self.routing_region or not self.routing_region.contains(Point(*self.center)):
            return "Blocked: Machine outside region"
        if any(machine_polygon.intersects(column) for column in self.columns):
            return "Blocked: Machine collides with column"
        if any(machine_polygon.intersection(region).area > 1e-7 for region in self.blocked_vertical_regions):
            return "Blocked: Insufficient vertical clearance for machine"
        return None

    def refresh_graph(self, pins):
        machine_polygon = self.polygon(pins)
        if self.graph_type != 0:
            self.build_grid(machine_pins=pins)
        self.update_dynamic_env(machine_polygon)

    def snap_pins(self, pins):
        if self.workspace.spatial_index is None:
            return {}
        targets = {}
        for spec in port_access_specs(self.spec, pins, self.angle):
            _, index = self.workspace.spatial_index.query(spec["access_point"])
            target = spec.copy()
            target["node_idx"] = int(index)
            targets.setdefault(spec["pin"], []).append(target)
        return targets
