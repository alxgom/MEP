"""Conservative raster screening for CLI body and supply extension placement."""

from dataclasses import dataclass
import math

import numpy as np
import shapely

from .geometry import cli_machine_pins, cli_placement_footprint_parts


PLACEMENT_MASK_STEP_MM = 50


@dataclass(frozen=True, eq=False)
class CliPlacementMask:
    origin_mm: tuple[int, int]  # lower-left tile edge
    step_mm: int
    visible: np.ndarray  # indexed [y, x], with Y increasing upwards
    valid: np.ndarray
    clear_room: np.ndarray  # valid and body inside one room without wall contact


def _footprint_windows(machine, quarter_turn):
    angle = quarter_turn * 90
    pins = cli_machine_pins(machine, (0, 0), angle)
    step = PLACEMENT_MASK_STEP_MM
    bounds = (part.bounds for part in cli_placement_footprint_parts(machine, pins, angle))
    # Signed inclusive tile offsets cover the footprint translated anywhere in its center tile.
    return tuple(
        (math.floor(left / step), math.floor(bottom / step),
         math.ceil(right / step), math.ceil(top / step))
        for left, bottom, right, top in bounds
    )


def _erode(free, windows):
    """Require every rectangle of the footprint to fit for the whole center tile."""
    ny, nx = free.shape
    counts = np.pad((~free).cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32), ((1, 0), (1, 0)))
    valid = np.ones_like(free)
    for left, bottom, right, top in windows:
        x0, x1 = max(0, -left), min(nx, nx - right)
        y0, y1 = max(0, -bottom), min(ny, ny - top)
        if x0 >= x1 or y0 >= y1:
            return np.zeros_like(free)
        a, b = slice(y0 + bottom, y1 + bottom), slice(y0 + top + 1, y1 + top + 1)
        c, d = slice(x0 + left, x1 + left), slice(x0 + right + 1, x1 + right + 1)
        valid[y0:y1, x0:x1] &= (counts[b, d] - counts[a, d] - counts[b, c] + counts[a, c]) == 0
        valid[:y0] = valid[y1:] = False
        valid[:, :x0] = valid[:, x1:] = False
    return valid


class CliPlacementMasks:
    """Rasterize the allowed region and clear rooms; cache by machine and rotation."""

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
            # Only the body must fit in one room for the placement to be green.
            rooms.append((window, wall_clear[window] & shapely.covers(area, tiles[window])))
        visible.setflags(write=False)
        self._rooms, self._visible, self._free = rooms, visible, free
        self._origin = origin_x, origin_y

    def get(self, dwelling, machine, angle_degrees):
        quarter_turn = round(angle_degrees / 90)
        if not math.isclose(angle_degrees, quarter_turn * 90, abs_tol=1e-7):
            return None
        if dwelling is not self._dwelling:
            self._rasterize(dwelling)
            self._dwelling = dwelling
        if self._visible is None:
            return None
        key = (machine, quarter_turn % 4)
        if key not in self._masks:
            body, extension = _footprint_windows(*key)
            valid = _erode(self._free, (body, extension))
            clear_room = np.zeros_like(valid)
            for window, free in self._rooms:
                clear_room[window] |= _erode(free, (body,))
            clear_room &= valid
            valid.setflags(write=False)
            clear_room.setflags(write=False)
            self._masks[key] = CliPlacementMask(self._origin, PLACEMENT_MASK_STEP_MM, self._visible, valid, clear_room)
        return self._masks[key]
