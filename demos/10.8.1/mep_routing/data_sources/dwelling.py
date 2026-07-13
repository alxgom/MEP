from __future__ import annotations

from shapely.geometry import LineString, MultiLineString


def iter_lines(geometry):
    """Yield line components from a Shapely geometry collection."""
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from iter_lines(part)


def cut_line_obstacles(line, columns, shafts):
    """Subtract structural columns and shafts from a wall centerline."""
    cut = line
    for column in columns:
        cut = cut.difference(column)
    for shaft in shafts:
        cut = cut.difference(shaft)
    return list(iter_lines(cut))


def derive_room_boundary_walls(rooms, columns, shafts, min_length_mm=50.0):
    """Derive shared room-boundary centerlines, excluding structural obstacles."""
    derived = []
    for index, room in enumerate(rooms):
        for other_room in rooms[index + 1:]:
            shared = room.polygon.boundary.intersection(other_room.polygon.boundary)
            for line in iter_lines(shared):
                if line.length > min_length_mm:
                    derived.extend(cut_line_obstacles(line, columns, shafts))
    return [line for line in derived if line.length > min_length_mm]


def build_wall_polygons(walls, columns, shafts, wall_thickness_mm):
    """Buffer wall centerlines and subtract structural obstacles."""
    wall_polygons = []
    for wall in walls:
        polygon = wall.buffer(float(wall_thickness_mm) / 2.0 - 0.1)
        for column in columns:
            polygon = polygon.difference(column)
        for shaft in shafts:
            polygon = polygon.difference(shaft)
        if not polygon.is_empty:
            wall_polygons.append(polygon)
    return wall_polygons


def choose_initial_machine_position(
    terminals,
    shaft_extraction,
    representative_point,
    fallback=(7500.0, 5500.0),
    priority_keywords=("Bathroom", "Washroom", "Toilet"),
):
    """Choose a terminal near the shaft, preferring wet-room terminal families."""
    if not terminals:
        return fallback
    if shaft_extraction is None:
        return next(iter(terminals.values()))

    shaft_x, shaft_y = representative_point(shaft_extraction)
    candidates = []
    for name, point in terminals.items():
        priority = 0 if any(keyword in name for keyword in priority_keywords) else 1
        candidates.append((priority, abs(shaft_x - point[0]) + abs(shaft_y - point[1]), point))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]
