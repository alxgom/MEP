from __future__ import annotations

import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep as shapely_prep

from mep_routing.geometry import point_segment_min_distances


def terminal_validity_entries(
    nodes,
    adj,
    terminal_names,
    routing_region,
    room_polygon_by_name,
    room_terminal_valid_region,
    room_terminal_boundary_segments,
    regulation_clearance_mm,
    terminal_buffer_mm,
):
    node_count = len(nodes)
    allowed_nodes = set()
    blocked_reasons = {}
    terminal_room_nodes = set()

    if routing_region is not None:
        for room_name in terminal_names:
            room_poly = room_polygon_by_name(room_name)
            if room_poly is None or room_poly.is_empty:
                continue

            valid_region = room_terminal_valid_region(room_name)
            if valid_region is None or valid_region.is_empty:
                continue

            prepared = shapely_prep(valid_region)
            room_node_indices = [
                int(i)
                for i, pt in enumerate(nodes)
                if adj.get(int(i)) and prepared.contains(Point(float(pt[0]), float(pt[1])))
            ]
            terminal_room_nodes.update(room_node_indices)
            if not room_node_indices:
                continue

            segments = room_terminal_boundary_segments(room_name)
            if len(segments) == 0:
                allowed_nodes.update(room_node_indices)
                continue

            pts = nodes[np.array(room_node_indices, dtype=np.int64)]
            distances = point_segment_min_distances(pts, segments)
            required_clearance = max(regulation_clearance_mm, terminal_buffer_mm)
            for node_idx, distance in zip(room_node_indices, distances):
                if float(distance) >= float(required_clearance) - 1e-7:
                    allowed_nodes.add(int(node_idx))
                    continue

                reasons = []
                if float(distance) < float(regulation_clearance_mm) - 1e-7:
                    reasons.append(f"inside {int(regulation_clearance_mm)} mm regulation clearance")
                if float(distance) < float(terminal_buffer_mm) - 1e-7:
                    reasons.append(f"inside {int(terminal_buffer_mm)} mm terminal buffer")
                if not reasons:
                    reasons.append(f"clearance below {int(required_clearance)} mm")
                blocked_reasons[int(node_idx)] = reasons

    entries = []
    reasons_by_node = {}
    for node_idx in range(node_count):
        pt = nodes[int(node_idx)]
        if int(node_idx) in allowed_nodes:
            allowed = True
            reasons = ["allowed terminal placement"]
        else:
            allowed = False
            if not adj.get(int(node_idx)):
                reasons = ["isolated graph node"]
            elif int(node_idx) in blocked_reasons:
                reasons = blocked_reasons[int(node_idx)]
            elif int(node_idx) not in terminal_room_nodes:
                reasons = ["outside terminal room"]
            else:
                reasons = ["blocked terminal placement"]
        entries.append((float(pt[0]), float(pt[1]), int(node_idx), allowed))
        reasons_by_node[int(node_idx)] = reasons

    return entries, reasons_by_node
