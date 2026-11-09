"""Conservative raster screening for CLI machine body placement."""

from dataclasses import dataclass
import math

import numpy as np
import shapely


PLACEMENT_MASK_STEP_MM = 50


@dataclass(frozen=True, eq=False)
class CliPlacementMask:
    origin_mm: tuple[int, int]  # lower-left tile edge
    step_mm: int
    visible: np.ndarray  # indexed [y, x], with Y increasing upwards
    valid: np.ndarray
    clear_room: np.ndarray  # valid and inside one room without wall contact


def _erode(free, radius_x, radius_y):
    height, width = 2 * radius_y + 1, 2 * radius_x + 1
    valid = np.zeros_like(free)
    if height <= free.shape[0] and width <= free.shape[1]:
        counts = np.pad((~free).cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32), ((1, 0), (1, 0)))
        blocked = counts[height:, width:] - counts[:-height, width:] - counts[height:, :-width] + counts[:-height, :-width]
        valid[radius_y:radius_y + blocked.shape[0], radius_x:radius_x + blocked.shape[1]] = blocked == 0
    return valid


class CliPlacementMasks:
    """Rasterize the allowed region and clear rooms; cache by oriented body size."""

    def __init__(self):
        self._dwelling = None
        self._rooms = ()
        self._visible = None
        self._free = None
        self._masks = {}

    def _rasterize(self, dwelling):
        self._rooms, self._visible, self._free = (), None, None
        self._masks.clear()
        step = PLACEMENT_MASK_STEP_MM
        region = dwelling.routing_region_base
        if region is None or region.is_empty:
            return
        x0, y0, x1, y1 = region.bounds
        origin_x, origin_y = math.floor(x0 / step) * step, math.floor(y0 / step) * step
        nx, ny = math.ceil((x1 - origin_x) / step), math.ceil((y1 - origin_y) / step)
        x = origin_x + np.arange(nx)[None, :] * step
        y = origin_y + np.arange(ny)[:, None] * step
        tiles = shapely.box(x, y, x + step, y + step)
        obstacles = shapely.union_all([*dwelling.columns, *dwelling.shafts])
        walls = shapely.union_all(dwelling.wall_polygons)
        shapely.prepare(region)
        shapely.prepare(obstacles)
        shapely.prepare(walls)
        visible = shapely.contains_xy(region, x + step / 2, y + step / 2)
        free = shapely.covers(region, tiles) & ~shapely.intersects(obstacles, tiles)
        # Interior walls classify a placement; the allowed region controls feasibility.
        wall_clear = free & ~shapely.intersects(walls, tiles)
        rooms = []
        for room in dwelling.rooms:
            area = region.intersection(room.polygon)
            if area.is_empty or not area.area:
                continue
            a, b, c, d = area.bounds
            i, j = math.floor((a - origin_x) / step), math.floor((b - origin_y) / step)
            k, l = math.ceil((c - origin_x) / step), math.ceil((d - origin_y) / step)
            window = np.s_[j:l, i:k]
            shapely.prepare(area)
            # A green footprint must fit in one room, even across a wall-free opening.
            rooms.append((window, wall_clear[window] & shapely.covers(area, tiles[window])))
        visible.setflags(write=False)
        self._rooms, self._visible, self._free = rooms, visible, free
        self._origin = origin_x, origin_y

    def get(self, dwelling, body_size_mm, angle_degrees):
        quarter_turn = round(angle_degrees / 90)
        if not math.isclose(angle_degrees, quarter_turn * 90, abs_tol=1e-7):
            return None
        if dwelling is not self._dwelling:
            self._rasterize(dwelling)
            self._dwelling = dwelling
        if self._visible is None:
            return None
        width, depth = body_size_mm[:2]
        size = (depth, width) if quarter_turn % 2 else (width, depth)
        if size not in self._masks:
            step = PLACEMENT_MASK_STEP_MM
            rx, ry = (math.ceil(value / (2 * step)) for value in size)
            valid = _erode(self._free, rx, ry)
            clear_room = np.zeros_like(valid)
            for window, free in self._rooms:
                clear_room[window] |= _erode(free, rx, ry)
            valid.setflags(write=False)
            clear_room.setflags(write=False)
            self._masks[size] = CliPlacementMask(self._origin, step, self._visible, valid, clear_room)
        return self._masks[size]
